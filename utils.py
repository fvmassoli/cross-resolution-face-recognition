import argparse


def parse_arguments():

    parser = argparse.ArgumentParser("LowResolutionFaceRecognition")

    # Generic usage
    parser.add_argument('-rs', '--seed', type=int, default=41, help='Set random seed (default: 41)')
    parser.add_argument('-g', '--useGpu', action='store_true', help='Use GPU (default: false)')
    parser.add_argument('-tm', '--testMode', action='store_true', help='Run in test mode (default: false)')
    parser.add_argument('-t', '--currentTime', help='Current time')

    # Dataset and run mode choices
    parser.add_argument('-dn', '--datasetName', choices=['tinyface', 'vggface2', 'vggface2-500', 'ijbb', 'ijbc', 'qmul'],
                        default='tinyface', help='Dataset name (default: tinyface)')
    parser.add_argument('-rm', '--runMode', choices=['train', 'extr_feat'],
                        default='extr_feat', help='Run mode (default: extr_feat)')

    # Model related options
    parser.add_argument('-mp', '--modelPath', help='Path to base model checkpoint')
    parser.add_argument('-lt', '--loadTeacherModel', action='store_true', help='Load teacher model (default: False)')
    parser.add_argument('-tp', '--modelTeacherPath', help='Path to teacher model')
    parser.add_argument('-ft', '--loadFineTunedModel', action='store_true', help='Load fine tuned model (default: false)')
    parser.add_argument('-ckp', '--modelCheckPointPath', help='Path to fine tuned model checkpoint')

    # Options to start train of a SeNet50 from scratch
    parser.add_argument('-se', '--loadSenet50', action='store_true', help='Load SeNet50 model (default: false)')
    parser.add_argument('-seft', '--pretrainedSenet50', action='store_true', help='Load SeNet50 model checkpoint (default: false)')
    parser.add_argument('-seckp', '--senet50CktPath', help='Path to fine SeNet50 checkpoint')

    # Use super resolved images for features extraction
    parser.add_argument('-sr', '--superResolvedImages', action='store_true',
                        help='Extract features from SR images. It is only valid for QMUL and Tinyface (default: False)')

    # Reset params options
    parser.add_argument('-rbn', '--resetBatchNormStats', action='store_true', help='Reset BatchNorm stats (default: False)')

    # Loss Options
    parser.add_argument('-ll', '--lossLambda', default=0.2, type=float,
                        help='Lambda for features regression loss (default: 0.2)')
    parser.add_argument('-lr', '--learningRate', default=0.001, type=float, help='Learning rate (default: 1.e-3)')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, help='Optimizer momentum (default: 0.9)')
    parser.add_argument('-lp', '--lowerResolutionProb', default=0.5, type=float,
                        help='Lowering resoltion probability (default: 0.5)')
    parser.add_argument('-e', '--trainingEpochs', type=int, default=1, help='Training epochs (default: 1)')
    parser.add_argument('-sv', '--stepsBeforeValidationRun', type=int, default=2,
                        help='Set number of iteration steps before each validation run (default: 2)')
    parser.add_argument('-s', '--useScheduler', action='store_true', help='Use scheduler (default: False)')
    parser.add_argument('-sp', '--schedulerPatience', type=int, default=20, help='Scheduler Patience (default: 20)')
    parser.add_argument('-fx', '--fixResolution', action='store_true', help='Use fix resolution while training (default: False)')
    parser.add_argument('-c', '--useCurriculumLearning', action='store_true', help='Use curriculum learning (default: False)')
    parser.add_argument('-cs', '--currStepIterations', type=int, default=35000, help='Number of images for each curriculum step (default: 35000)')

    # Layers Options
    parser.add_argument('-ud', '--useDropOut', action='store_true', help='Use DropOut (default: False)')
    parser.add_argument('-dp', '--dropOutProb', default=0.5, type=float, help='DropOut probability (default: 0.5)')
    parser.add_argument('-fn', '--freezeNet', action='store_true', help='Freeze params (default: False)')
    parser.add_argument('-rfc', '--resetFullyConnected', action='store_true', help='Reset final FC layer (default: False)')

    # Added ontly to downsample scface images at 64 pixels
    parser.add_argument('-ds', '--downsample', action='store_true')

    return parser.parse_args()
