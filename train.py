import argparse, json, os
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import CIFAR10_captioning
from decoder import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy, calculate_caption_lengths


'''
=== Arguments ===
'''

parser = argparse.ArgumentParser(description='Show, Attend and Tell')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train for (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of the decoder (default: 1e-4)')
parser.add_argument('--step-size', type=int, default=5,
                    help='step size for learning rate annealing (default: 5)')
parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                    help='regularization constant (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                    help='number of batches to wait before logging training stats (default: 100)')
parser.add_argument('--data', type=str, default='data/coco',
                    help='path to data images (default: data/coco)')
parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='vgg19',
                    help='Network to use in the encoder (default: vgg19)')
parser.add_argument('--model', type=str, help='path to model', default=None)
parser.add_argument('--tf', action='store_true', default=False,
                    help='Use teacher forcing when training LSTM (default: False)')

# Custom arguments
parser.add_argument('--gpu-id', type=int, default=0, required=True)
parser.add_argument('--data-download', action='store_true', default=False)
parser.add_argument('--pretrained-encoder', action='store_true', default=False,
                    help='if this arg is True, use pretrained weight for encoder')
parser.add_argument('--ckpt-dir', type=str, required=True)


# Parse arguments
args = parser.parse_args()

'''
================
'''


# Set GPU
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main(args):
    writer = SummaryWriter()

    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(network=args.network, pretrained=args.pretrained_encoder)
    decoder = Decoder(vocabulary_size, encoder.dim, args.tf)

    if args.model:
        decoder.load_state_dict(torch.load(args.model))

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    data_dir_path = args.data

    trainset = CIFAR10_captioning(root=data_dir_path, train=True, download=args.data_download, transform = data_transforms, token2idx = word_dict)
    testset = CIFAR10_captioning(root=data_dir_path, train=False, download=args.data_download, transform = data_transforms, token2idx = word_dict)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=True, num_workers=4)

    print('Starting training with {}'.format(args))

    model_dir = os.path.join('fintuned_models/', args.ckpt_dir)
    if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    best_top1_acc = 0.

    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        train(epoch, args.pretrained_encoder, encoder, decoder, optimizer, cross_entropy_loss,
              train_loader, word_dict, args.alpha_c, args.log_interval, writer)
        top1_acc = validate(epoch, encoder, decoder, cross_entropy_loss, val_loader,
                 word_dict, args.alpha_c, args.log_interval, writer)
        model_file = os.path.join(model_dir, 'model_' + args.network + '_' + str(epoch) + '.pth')
        torch.save(decoder.state_dict(), model_file)
        print('Saved model to ' + model_file)

        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            torch.save(decoder.state_dict(), os.path.join(model_dir, 'best_' + args.network + '.pth'))
        
    writer.close()


def train(epoch, is_pretrained, encoder, decoder, optimizer, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer):
    if is_pretrained:
        encoder.eval()
    else:
        encoder.train()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (imgs, (captions, captions_ids)) in enumerate(data_loader):
        imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
        img_features = encoder(imgs)
        optimizer.zero_grad()

        preds, alphas = decoder(img_features, captions)
        targets = captions[:, 1:]

        targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
        preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]

        att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss.backward()
        optimizer.step()

        total_caption_length = calculate_caption_lengths(word_dict, captions)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
    writer.add_scalar('train_loss', losses.avg, epoch)
    writer.add_scalar('train_top1_acc', top1.avg, epoch)
    writer.add_scalar('train_top5_acc', top5.avg, epoch)


def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict, alpha_c, log_interval, writer):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # used for calculating bleu scores
    references = []
    hypotheses = []

    with torch.no_grad():
        for batch_idx, (imgs, (captions, captions_ids)) in enumerate(data_loader):
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
            img_features = encoder(imgs)
            preds, alphas = decoder(img_features, captions)
            targets = captions[:, 1:]
            
            packed_targets = pack_padded_sequence(targets, [len(tar) - 1 for tar in targets], batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, [len(pred) - 1 for pred in preds], batch_first=True)[0]
            
            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(packed_preds, packed_targets)
            loss += att_regularization
            
            total_caption_length = calculate_caption_lengths(word_dict, captions)
            acc1 = accuracy(preds, targets, 1)
            acc5 = accuracy(preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            cap = []
            for caption in captions:
                cap.append( [ [word_idx for word_idx in caption
                                if word_idx != word_dict['<start>'] and word_idx != word_dict['<pad>']] ] )
            references += cap

            word_idxs = torch.max(preds, dim=2)[1]
            for idxs in word_idxs.tolist():
                hypotheses.append([idx for idx in idxs
                                       if idx != word_dict['<start>'] and idx != word_dict['<pad>']])

            if batch_idx % log_interval == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))
                          
        writer.add_scalar('val_loss', losses.avg, epoch)
        writer.add_scalar('val_top1_acc', top1.avg, epoch)
        writer.add_scalar('val_top5_acc', top5.avg, epoch)
        
        bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
        bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
        bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
        bleu_4 = corpus_bleu(references, hypotheses)

        writer.add_scalar('val_bleu1', bleu_1, epoch)
        writer.add_scalar('val_bleu2', bleu_2, epoch)
        writer.add_scalar('val_bleu3', bleu_3, epoch)
        writer.add_scalar('val_bleu4', bleu_4, epoch)
        print('Validation Epoch: {}\t'
             'BLEU-1 ({})\t'
             'BLEU-2 ({})\t'
             'BLEU-3 ({})\t'
             'BLEU-4 ({})\t'.format(epoch, bleu_1, bleu_2, bleu_3, bleu_4))

    return float(top1.val)


if __name__ == "__main__":
    main(args)