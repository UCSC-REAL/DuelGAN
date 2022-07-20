from gan_training.models import (dcgan_deep, dcgan_shallow,dcgan_deep32)

generator_dict = {
    'dcgan_deep': dcgan_deep.Generator,
    'dcgan_deep32':dcgan_deep32.Generator,
    'dcgan_shallow': dcgan_shallow.Generator
}

discriminator_dict = {
    'dcgan_deep': dcgan_deep.Discriminator,
    'dcgan_deep32': dcgan_deep32.Discriminator,
    'dcgan_shallow': dcgan_shallow.Discriminator
}
