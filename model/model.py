# @title Default title text
import torch
from deform_cnn.Deform_conv import DeformableConv2d

class Encoder_mod(torch.nn.Module):
    def __init__(self, num_classes=102, num_filters=32, input_size=(16, 1,40,40)):
        super(Encoder_mod, self).__init__()

        kernel_size = (5, 1)



        # Encoder Layer1
        self.encoder_layer1_conv = torch.nn.Conv2d(1, num_filters, 5,1, padding='same')# ,padding=(3, 3))
        self.encoder_layer1_batch_norm = torch.nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.99)
        self.encoder_layer1_pooling = torch.nn.AvgPool2d((2, 2))
        self.encoder_layer1_activation = torch.nn.ReLU()
       # Encoder Layer2
        self.encoder_layer2_conv = torch.nn.Conv2d(num_filters, 2 * num_filters, 5,1, padding='same')# ,padding=(3, 3))
        self.encoder_layer2_activation = torch.nn.ReLU()
        self.encoder_layer2_batch_norm = torch.nn.BatchNorm2d(2 * num_filters, eps=1e-3, momentum=0.99)
        self.encoder_layer2_pooling = torch.nn.AvgPool2d((2, 2))
        # Encoder Layer3
        self.encoder_layer3_conv = torch.nn.Conv2d(2 * num_filters, 3*num_filters, 5,1, padding='same') # ,padding=(3, 3))
        self.encoder_layer3_batch_norm = torch.nn.BatchNorm2d(3*num_filters, eps=1e-3, momentum=0.99)
        self.encoder_layer3_pooling = torch.nn.AvgPool2d((2, 2))
        self.encoder_layer3_activation = torch.nn.ReLU()
        #LAYER 4
        self.flat=torch.nn.Flatten()
        self.fc1_linear1 = torch.nn.Linear(2400, num_classes)
        self.fc1_linear2 = torch.nn.Linear(num_classes, num_classes)
        self.fc1_activation = torch.nn.ReLU()
        self.drop = torch.nn.Dropout()
    def forward(self, x):
        # torch.nn.init.xavier_uniform_(self.encoder_layer1_conv.weight)
        # torch.nn.init.xavier_uniform_(self.encoder_layer2_conv.weight)
        # torch.nn.init.xavier_uniform_(self.encoder_layer3_conv.weight)
        # torch.nn.init.xavier_uniform_(self.fc1_linear1.weight)
        # torch.nn.init.xavier_uniform_(self.fc1_linear2.weight)

        x = self.encoder_layer1_conv(x)
        x = self.encoder_layer1_batch_norm (x)
        x= self.encoder_layer1_pooling(x)
        x = self.encoder_layer1_activation(x)
        x = self.encoder_layer2_conv(x)
        #x=self.drop(x)
        x = self.encoder_layer2_activation(x)
        x= self.encoder_layer2_batch_norm(x)
        x = self.encoder_layer2_pooling(x)
        x= self.encoder_layer3_conv(x)
        #x=self.drop(x)

        x = self.encoder_layer3_activation(x)
        x = self.encoder_layer3_pooling(x)
        x= self.flat(x)
        x=self.fc1_linear1(x)

        encoded_output= self.fc1_activation(x)
        encoded_output=self.fc1_linear2(encoded_output)



        return encoded_output

class Decoder_mod(torch.nn.Module):
    def __init__(self, num_classes=102, num_filters=32, input_size=(16, 1,40,40)):
        super(Decoder_mod, self).__init__()
        #Decoder
        self.decoder_fc1_linear1 = torch.nn.Linear(102,2400)
        self.decoder_layer1_conv =  torch.nn.Conv2d(3 * num_filters, 3*num_filters, 5,1, padding='same')
        self.decoder_layer1_batch_norm = torch.nn.BatchNorm2d( 3*num_filters, eps=1e-3, momentum=0.99)
        self.decoder_layer1_pooling = torch.nn.Upsample(scale_factor=2,mode='bilinear')
        self.decoder_layer2_conv =torch.nn.Conv2d(3*num_filters, 2*num_filters, 5,1, padding='same')
        self.decoder_layer2_batch_norm = torch.nn.BatchNorm2d(2*num_filters, eps=1e-3, momentum=0.99)
        self.decoder_layer3_conv =torch.nn.Conv2d(2*num_filters , num_filters, 5,1, padding='same')
        self.decoder_layer3_batch_norm = torch.nn.BatchNorm2d(num_filters, eps=1e-3, momentum=0.99)
        self.activation = torch.nn.ReLU()
        self.activation1 = torch.nn.ReLU()
        self.decoder_layer4_conv =  torch.nn.Conv2d(num_filters,1 , 5,1, padding='same')
        self.upsample = torch.nn.Upsample(size=(1, 2000), mode='bilinear', align_corners=False)
        self.linear2= torch.nn.Linear(40,40)
    def forward(self, x):
        x=x.unsqueeze(1).unsqueeze(2)
        x=self.decoder_fc1_linear1(x)
        x=x.view((-1,96,5,5))

        x = self.decoder_layer1_conv(x)
        x=self.activation(x)
        x1=self.decoder_layer1_pooling(x)
        x = self.decoder_layer1_batch_norm(x1)

        x = self.decoder_layer2_conv(x)
        x = self.activation(x)
        x1=self.decoder_layer1_pooling(x)
        x = self.decoder_layer2_batch_norm(x1)

        x = self.decoder_layer3_conv(x)
        x = self.activation(x)
        x1=self.decoder_layer1_pooling(x)
        x=self.decoder_layer4_conv(x1)
        decoded_output=self.activation(x)
        return    decoded_output

class LSE(torch.nn.Module):
    def __init__(self, num_classes=102, num_filters=32, input_size=(16, 1,40,40)):
        super(LSE, self).__init__()
        self.feature_extraction_module = Encoder_mod(num_classes=num_classes, num_filters=num_filters)
        self.feature_decoder_module = Decoder_mod(num_classes=num_classes, num_filters=num_filters)
    def forward(self,x):
          encoded_output=  self.feature_extraction_module(x)
          decoded_output=self.feature_decoder_module(encoded_output)
          return decoded_output,encoded_output
