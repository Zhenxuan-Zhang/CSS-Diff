# This code is released under the CC BY-SA 4.0 license.

import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from models.pss_generator import PSSGenerator
from torchmetrics.image import StructuralSimilarityIndexMeasure
class PSSModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        return parser


    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B','ssim','L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']
            
        self.L1_loss = torch.nn.L1Loss()
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = PSSGenerator(config=opt.PSS, ckpt_path=None, device=opt.device).to(opt.device)
        self.netG_B = PSSGenerator(config=opt.PSS, ckpt_path=None, device=opt.device).to(opt.device)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.PSS['optim']['beta1'], 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device).float()
        self.real_B = input['B' if AtoB else 'A'].to(self.device).float()
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     self.t = torch.randint(0, 100, (self.real_A.shape[0],), device=self.device)
    #     self.B_t, self.noise_0 = self.netG_A.q_sample(self.real_B,self.t)
    #     self.fake_B = self.netG_A(self.B_t, cond=self.real_A)  # G_A(A)
        
    #     self.rec_A = self.netG_B(self.fake_B, cond=self.fake_B)
    #     #self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        
    #     self.A_t, self.noise_1 = self.netG_B.q_sample(self.real_A,self.t)
    #     self.fake_A = self.netG_B(self.A_t,cond=self.real_B)  # G_B(B)
        
    #     #self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
    #     self.rec_B = self.netG_A(self.fake_A, cond=self.fake_A)
    def forward(self):
        # print(f"real_A: {self.real_A.device}, real_B: {self.real_B.device}")
        self.t = torch.randint(0, 50, (self.real_A.shape[0],), device=self.device)
        # print(f"t: {self.t.device}")
        self.B_t, self.noise_0 = self.netG_A.q_sample(self.real_A, self.t)
        #print(f"B_t: {self.B_t.device}, noise_0: {self.noise_0.device}")
        self.fake_B = self.netG_A(self.B_t, cond=self.real_A)
        #print(f"fake_B: {self.fake_B.device}")
        self.B_rec, self.noise_0rec = self.netG_B.q_sample(self.fake_B, self.t)
        self.rec_A = self.netG_B(self.B_rec, cond=self.fake_B)
        #print(f"rec_A: {self.rec_A.device}")
        self.A_t, self.noise_1 = self.netG_B.q_sample(self.real_B, self.t)
        #print(f"A_t: {self.A_t.device}, noise_1: {self.noise_1.device}")
        self.fake_A = self.netG_B(self.A_t, cond=self.real_B)
        #print(f"fake_A: {self.fake_A.device}")
        self.A_rec, self.noise_1rec = self.netG_A.q_sample(self.fake_A, self.t)
        self.rec_B = self.netG_A(self.A_rec, cond=self.fake_A)
        #print(f"rec_B: {self.rec_B.device}")
    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_L1 = self.opt.lambda_L1
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # self.idt_A = self.netG_A(self.real_B)
            # self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.idt_B = self.netG_B(self.real_A)
            # self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
            # self.t_idt = torch.randint(0, 50, (self.real_B.size(0),), device=self.device)
            #self.t_idt_B = torch.randint(0, 50, (self.real_A.size(0),), device=self.device)

            # Add noise at t
            # self.real_B_noisy, _ = self.netG_A.q_sample(self.real_B, self.t_idt)
            # self.real_A_noisy, _ = self.netG_B.q_sample(self.real_A, self.t_idt)

            # Try to recover real_B (identity mapping)
            self.idt_A = self.fake_A#self.netG_A(self.real_B_noisy, cond=self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt

            # Try to recover real_A (identity mapping)
            self.idt_B = self.fake_B#self.netG_B(self.real_A_noisy, cond=self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_L1 = self.L1_loss(self.real_B,self.fake_B)*lambda_L1
        # Ensure images are in [0, 1] range for SSIM
        real_B_clamped = (self.real_B + 1) / 2
        fake_B_clamped = (self.fake_B + 1) / 2

        ssim_score = self.ssim_metric(fake_B_clamped, real_B_clamped)
        self.loss_ssim = (1 - ssim_score) * lambda_B  # lower SSIM → higher loss

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B +self.loss_L1 + self.loss_ssim
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights