import streamlit as st
import numpy as np
import json
import random
import cv2
import torch
from PIL import Image
from collections import Counter

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor


classes = ['1_puffed_food',
           '2_puffed_food',
           '3_puffed_food',
           '4_puffed_food',
           '5_puffed_food',
           '6_puffed_food',
           '7_puffed_food',
           '8_puffed_food',
           '9_puffed_food',
           '10_puffed_food',
           '11_puffed_food',
           '12_puffed_food',
           '13_dried_fruit',
           '14_dried_fruit',
           '15_dried_fruit',
           '16_dried_fruit',
           '17_dried_fruit',
           '18_dried_fruit',
           '19_dried_fruit',
           '20_dried_fruit',
           '21_dried_fruit',
           '22_dried_food',
           '23_dried_food',
           '24_dried_food',
           '25_dried_food',
           '26_dried_food',
           '27_dried_food',
           '28_dried_food',
           '29_dried_food',
           '30_dried_food',
           '31_instant_drink',
           '32_instant_drink',
           '33_instant_drink',
           '34_instant_drink',
           '35_instant_drink',
           '36_instant_drink',
           '37_instant_drink',
           '38_instant_drink',
           '39_instant_drink',
           '40_instant_drink',
           '41_instant_drink',
           '42_instant_noodles',
           '43_instant_noodles',
           '44_instant_noodles',
           '45_instant_noodles',
           '46_instant_noodles',
           '47_instant_noodles',
           '48_instant_noodles',
           '49_instant_noodles',
           '50_instant_noodles',
           '51_instant_noodles',
           '52_instant_noodles',
           '53_instant_noodles',
           '54_dessert',
           '55_dessert',
           '56_dessert',
           '57_dessert',
           '58_dessert',
           '59_dessert',
           '60_dessert',
           '61_dessert',
           '62_dessert',
           '63_dessert',
           '64_dessert',
           '65_dessert',
           '66_dessert',
           '67_dessert',
           '68_dessert',
           '69_dessert',
           '70_dessert',
           '71_drink',
           '72_drink',
           '73_drink',
           '74_drink',
           '75_drink',
           '76_drink',
           '77_drink',
           '78_drink',
           '79_alcohol',
           '80_alcohol',
           '81_drink',
           '82_drink',
           '83_drink',
           '84_drink',
           '85_drink',
           '86_drink',
           '87_drink',
           '88_alcohol',
           '89_alcohol',
           '90_alcohol',
           '91_alcohol',
           '92_alcohol',
           '93_alcohol',
           '94_alcohol',
           '95_alcohol',
           '96_alcohol',
           '97_milk',
           '98_milk',
           '99_milk',
           '100_milk',
           '101_milk',
           '102_milk',
           '103_milk',
           '104_milk',
           '105_milk',
           '106_milk',
           '107_milk',
           '108_canned_food',
           '109_canned_food',
           '110_canned_food',
           '111_canned_food',
           '112_canned_food',
           '113_canned_food',
           '114_canned_food',
           '115_canned_food',
           '116_canned_food',
           '117_canned_food',
           '118_canned_food',
           '119_canned_food',
           '120_canned_food',
           '121_canned_food',
           '122_chocolate',
           '123_chocolate',
           '124_chocolate',
           '125_chocolate',
           '126_chocolate',
           '127_chocolate',
           '128_chocolate',
           '129_chocolate',
           '130_chocolate',
           '131_chocolate',
           '132_chocolate',
           '133_chocolate',
           '134_gum',
           '135_gum',
           '136_gum',
           '137_gum',
           '138_gum',
           '139_gum',
           '140_gum',
           '141_gum',
           '142_candy',
           '143_candy',
           '144_candy',
           '145_candy',
           '146_candy',
           '147_candy',
           '148_candy',
           '149_candy',
           '150_candy',
           '151_candy',
           '152_seasoner',
           '153_seasoner',
           '154_seasoner',
           '155_seasoner',
           '156_seasoner',
           '157_seasoner',
           '158_seasoner',
           '159_seasoner',
           '160_seasoner',
           '161_seasoner',
           '162_seasoner',
           '163_seasoner',
           '164_personal_hygiene',
           '165_personal_hygiene',
           '166_personal_hygiene',
           '167_personal_hygiene',
           '168_personal_hygiene',
           '169_personal_hygiene',
           '170_personal_hygiene',
           '171_personal_hygiene',
           '172_personal_hygiene',
           '173_personal_hygiene',
           '174_tissue',
           '175_tissue',
           '176_tissue',
           '177_tissue',
           '178_tissue',
           '179_tissue',
           '180_tissue',
           '181_tissue',
           '182_tissue',
           '183_tissue',
           '184_tissue',
           '185_tissue',
           '186_tissue',
           '187_tissue',
           '188_tissue',
           '189_tissue',
           '190_tissue',
           '191_tissue',
           '192_tissue',
           '193_tissue',
           '194_stationery',
           '195_stationery',
           '196_stationery',
           '197_stationery',
           '198_stationery',
           '199_stationery',
           '200_stationery']

# Put target classes in alphabetical order (required for the labels being generated)
#classes.sort()

# Set up default variables

CONFIG_FILE = "COCO-Detection-faster_rcnn_R_101_FPN_3x/config_global.yaml"
MODEL_FILE = "COCO-Detection-faster_rcnn_R_101_FPN_3x/model_final.pth"

#DIC_TEST = "COCO-Detection-faster_rcnn_R_101_FPN_3x/rpc_metadata_test.pkl"

# TODO Way to load model with @st.cache so it doesn't take a long time each time
@st.cache(allow_output_mutation=True)
def create_predictor(model_config, model_weights, threshold):
    """
    Loads a Detectron2 model based on model_config, model_weights and creates a default
    Detectron2 predictor.

    Returns Detectron2 default predictor and model config.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_config)
    #cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = "cpu"
  
    cfg.MODEL.WEIGHTS = model_weights
  
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model

    cfg.DATASETS.TEST = ("rpc_tst_2019_dataset",)

    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0

    predictor = DefaultPredictor(cfg)

    return cfg, predictor


@st.cache(allow_output_mutation=True)
def make_inference(predictor, image):
    img = np.asarray(image)
    return predictor(img[:, :, ::-1])


@st.cache(allow_output_mutation=True)
def output_image(cfg, img, outputs):

    img = np.asarray(img)

    v = Visualizer(img, MetadataCatalog.get(cfg.DATASETS.TEST[0]).set(thing_classes=classes), scale=2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
 
    processed_img = cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)


    return processed_img, outputs["instances"]


def main():
    st.title("Large-Scale Retail Product Detection 👁")
    st.write("This application detects Chinese products in automatic checkout [RPC: A Large-Scale Retail Product Checkout Dataset](https://www.kaggle.com/diyer22/retail-product-checkout-dataset).")
    st.write("You can download test images from this my Github repo: [Clutter test images](https://github.com/feryah/product-detection-detectron2/tree/master/sample_images_test_for_inference).")
    st.write("## How does it work?")
    st.write("Add an image of a products clutter and a Deep Learning model will look at it and find the products names like the example below:")

    st.image(Image.open("images/FotoJet.jpg"), 
             caption="Example of model being run on a products clutter.", 
             use_column_width=True)
    st.write("## Upload your own image")
    st.write("**Note:** The model has been trained on specific Chinese retail products and therefore will only work with those kind of images.") # Also, as the model has been trained on a sample of data, it might not work perfectly. This will be improved in the future.
    uploaded_image = st.file_uploader("Choose a png or jpg image", 
                                      type=["jpg", "png", "jpeg"])

    cfg, predictor = create_predictor(model_config=CONFIG_FILE, model_weights=MODEL_FILE, threshold=0.5)

    if uploaded_image is not None:
 

      image = Image.open(uploaded_image)

      image = image.convert("RGB")

      file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type,"FileSize":uploaded_image.size}


      st.image(image, caption="Uploaded Image", use_column_width=True)
      st.write(file_details)

      with st.spinner("Doing the math..."):
        outputs = make_inference(predictor, image)
        out_image, preds = output_image(cfg, image, outputs)
        st.image(out_image, caption="Processed Image", use_column_width=True) 


      st.write("Products detected:")
      st.write(dict(Counter([classes[i] for i in np.array(preds.pred_classes)])))
       
        
    st.write("## How is this made?")
    st.write("The Deep Learning happens with a fine-tuned [Detectron2](https://detectron2.readthedocs.io/) model (PyTorch), \
    this front end (what you're reading) is built with  \
    and it's all hosted on [Streamlit](https://www.streamlit.io/).") #[Google's App Engine](https://cloud.google.com/appengine/)
    st.write("### ~ Ferial Y.")
    #st.write("See the [code on GitHub](https://github.com/mrdbourke/airbnb-object-detection) and a [YouTube playlist](https://www.youtube.com/playlist?list=PL6vjgQ2-qJFeMrZ0sBjmnUBZNX9xaqKuM) detailing more below.")
    #st.video("https://youtu.be/C_lIenSJb3c")

if __name__ == "__main__":
    main()