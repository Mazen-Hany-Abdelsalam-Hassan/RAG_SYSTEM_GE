from fitz import open
import os
from src.models import DetectionModel
from src.utils import chunk_the_page ,visualize_bbox
from src.config import  (DEBUG_DIR , ID_TO_NAMES ,CONF_THRESHOLD,
                         IOU_THRESHOLD,TEXT,TEXT_IMAGE,TABLE)
from PIL import Image


class Document:
    def __init__(self, pdf_directory, use_agentic_chunker = None):
        self._PDF = open(pdf_directory)
        self._Detection_model = DetectionModel()
        #self.Table_parser = Table_parser
        self._Content = {}
        self._counter = 0
        pdf_dir = os.path.split(pdf_directory)[1].split('.')[0]
        os.makedirs(DEBUG_DIR, exist_ok=True)
        self.pdf_test = os.path.join(DEBUG_DIR, pdf_dir)
        os.makedirs(self.pdf_test, exist_ok=True)
    def extract_the_pdf_content(self):
        """this function apply the layout parser to chunk the documents
        this functions updates self._pdf_content_image
        """
        if self._counter >=1 :
            return

        for i in range(len(self._PDF)):
            image = self._PDF[i].get_pixmap(dpi=400).pil_image()
            #image = image.resize((800,1024))
            bboxes , score, classes = self._Detection_model.recognize_image(image,
                                        conf_threshold=CONF_THRESHOLD,
                                        iou_threshold=IOU_THRESHOLD)

            _,sorted_index= bboxes.sort(axis=0)
            bboxes = bboxes[sorted_index[::, 1]]
            score  = score[sorted_index[::, 1]]
            classes = classes[sorted_index[::, 1]]
            result_viz = Image.fromarray(visualize_bbox(image,bboxes,classes,
                                                        score,ID_TO_NAMES))
            save_dir = os.path.join(self.pdf_test,f"{i}.jpg")
            result_viz.save(save_dir)

            text_chunks, image_chunks , table_chunk = chunk_the_page(image, bboxes, classes)
            Temp_dict = {TEXT_IMAGE :image_chunks , TEXT:text_chunks,
                         TABLE:table_chunk}
            self._Content[f'Page{i}'] = Temp_dict

            #if i == 3:
             #   return bboxes, score, classes


        self._counter+=1

    def show(self):
        return self._Content





