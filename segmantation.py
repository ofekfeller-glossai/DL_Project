import pixellib
from pixellib.torchbackend.instance import instanceSegmentation


class Segmentation:
    model = r"data/models/model.pkl"
    default_output_folder = r"data/outputs"
    default_output = fr"{default_output_folder}/default.jpg"

    def __init__(self):
        self.ins_seg = instanceSegmentation()
        self.ins_seg.load_model(self.model)

    def segment_photo(self, image_path, output_path=default_output,extract_object=True):

        self.ins_seg.segmentImage(image_path,
                                  extract_segmented_objects=extract_object,
                                  save_extracted_objects=extract_object,
                                  show_bboxes=True,
                                  output_image_name=output_path)


def test():
    image = r"data/images/input_images/couple.jpeg"

    seg_model = Segmentation()

    seg_model.segment_photo(image)

if __name__ == '__main__':
    test()
