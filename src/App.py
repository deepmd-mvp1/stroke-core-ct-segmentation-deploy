from flask import Flask, flash, request, redirect, render_template
import tempfile
from flask_cors import CORS
import os
import json
from socket import gethostname
from niclib.architecture.SUNet import SUNETx5
from niclib.network.generator import *
from niclib.network.optimizers import TorchOptimizer
from niclib.network.loss_functions import *
from niclib.network.training import EarlyStoppingTrain
from niclib.evaluation.prediction import PatchPredictor
from niclib.evaluation.testing import TestingPrediction
from niclib.utils import *
torch.set_default_tensor_type('torch.FloatTensor')
from DataTXT import DataTXT # It works, no worries
from evaluation import DockerValidation
from art import tprint

app=Flask(__name__)
CORS(app)
app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 * 1024

# Get current path
path = os.getcwd()
params=None
# file Upload
UPLOAD_FOLDER = "/home/input"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

models_dict_pathfile = os.path.join('./', 'models.txt')
with open(models_dict_pathfile, 'r') as models_file:
    pretrained_dict = json.loads(models_file.read())  # use `json.dumps` to do the reverse

# Load path and associated optionsS
pretrained_path = pretrained_dict[params['pretrained_name']]['path']
pretrained_sym = pretrained_dict[params['pretrained_name']]['symmetric_modalities']
pretrained_num_mods = pretrained_dict[params['pretrained_name']]['num_modalities']
model_def = torch.load(pretrained_path)
model_parameters = filter(lambda p: p.requires_grad, model_def.parameters())
nparams = sum([np.prod(p.size()) for p in model_parameters])
visible_dict = {}
os.environ['CUDA_VISIBLE_DEVICES'] = visible_dict.get(gethostname(), '0')  # Return '0' by default
patch_shape_pred = (64, 64, 1)

predictor = PatchPredictor(
        num_classes=2,
        lesion_class=1,
        uncertainty_passes=3,
        uncertainty_dropout=0.1,
        zeropad_shape=patch_shape_pred,
        instruction_generator=PatchGeneratorBuilder(
            batch_size=256,
            shuffle=False,
            zeropad_shape=None,
            instruction_generator=PatchInstructionGenerator(
                in_shape=patch_shape_pred,
                out_shape=patch_shape_pred,
                sampler=UniformSampling(
                    in_shape=patch_shape_pred, extraction_step=(3, 3, 1)),
                augment_to=None)))



@app.route('/stroke/test', methods=['GET'])
def upload_form():
    return render_template('upload.html')


    
@app.route('/stroke/predict', methods=['POST'])
def Prediction():
   
    # os.mkdir(app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':

        files = request.files.getlist('files[]')
        inputDir = tempfile.mkdtemp(dir="input")
        results_path = tempfile.mkdtemp(dir="output")
        print("input file + " + inputDir)
        for file in files:
            filename = secure_filename(file.filename)
            print(filename)
            file.save(inputDir +"/" +filename)
            test_pred = TestingPrediction(predictor=predictor, out_path=results_path, save_probs=True, save_seg=False)
            test_pred.predict_test_set(model_def, files)
            result = infer(inputDir,filename)
            
            return send_file(inputDir +"/" + "pred.jpg", mimetype="image/jpg")



if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=False,threaded=True)