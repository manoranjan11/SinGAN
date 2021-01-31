from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions


if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()
    # post_config is from SinGAN.functions
    # Configuration initialization
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    # generate_dir2save is from SinGAN.functions
    # Generate relevant directories
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        # read_image is from SinGAN.functions
        # reads an image and outputs a torch tensor
        real = functions.read_image(opt)
        # adjust_scales2image is from SinGAN.functions
        # resize image to scale
        functions.adjust_scales2image(real, opt)
        # train is from SinGAN.training
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
