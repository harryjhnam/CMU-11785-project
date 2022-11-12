import os
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, encoder_output_size, decoder_output_size):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder # VGG
        self.decoder = decoder # LSTM

        self.affine = nn.Linear(encoder_output_size, decoder_output_size)

    def forward(self, x):
        image_embedding = self.encoder(x) # (batch_size, encoder_output_size)
        output = self.affine(image_embedding) # (batch_size, decoder_output_size)
        output = output.unsqueeze(1) # (batch_size, 1, decoder_output_size)

        target_length = 6
        for t in range(target_length):
            prev_state = None
            output, prev_state = self.decoder(output, prev_state)
            # output = (batch_size, 1, decoder_output_size)
            if t == 0:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=1)
        
        # outputs = (batch_size, target_legnth, decoder_output_size)

        return outputs