## Pytorch Implementation of [Zero-TextCap: Zero-shot Framework for Text-based Image Captioning][MM 2023]
***
## DEMO

### Preparation

#### Dataset preparation:
Please download TextCaps dataset (https://textvqa.org/textcaps/) and put them under the project.

Please download [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) and [BERT](https://huggingface.co/bert-base-uncased) from Huggingface Space.
#### OCR preparation:
Due to continuous iterations of the Google OCR API, we strongly recommend using Google OCR (https://cloud.google.com/vision/docs/ocr?hl=zh-cn) to re-extract text blocks and their contents. 

#### Data preparation:
After you have completed the OCR text extraction, please save the validation and test sets according to the format specified in "./textcaps/textcaps_val_label" in the "./textcaps/" directory.

|-image_id
|-ocr_tokens
|-ocr_normalized_boxes
|-features

'features' represents the features of the object from clip image encoder. Concretely, we select the object with smallest bounding box that contains the text block.
#### Environments preparation:
```
pip install -r requirements.txt
```
****
<!--### Citation
Please cite our work if you use it in your research:
```
@inproceedings{xu2023zero,
  title={Zero-TextCap: Zero-shot Framework for Text-based Image Captioning},
  author={Xu, Dongsheng and Zhao, Wenye and Cai, Yi and Huang, Qingbao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4949--4957},
  year={2023}
}
```
### Acknowledgment 
This code is based on the [bert-gen](https://github.com/nyu-dl/bert-gen) and [ConZIC](https://github.com/joeyz0z/ConZIC). -->


