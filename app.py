# Import the Dependencies
from transformers import pipeline
import os

# Function to remove the background from the image
def AI_Background_Remover(
        image_folder:str, 
        image_name:str, 
        result_image_folder:str, 
        result_image_name:str
        ):
    
    # Image path
    image_path = os.path.join(image_folder,image_name)

    # Use the transformer library pipline
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device="cpu") # You can use "cuda" also.

    # Load the image
    pillow_mask = pipe(image_path, return_mask = True) # outputs a pillow mask

    pillow_image = pipe(image_path) # applies mask on input and returns a pillow image

    # Save the mask
    pillow_mask.save(os.path.join("segmented_mask", f"{image_name[:-4]}_segmented_mask.png"))

    # Save the image
    pillow_image.save(os.path.join(result_image_folder, result_image_name))

# To Remove specific images background.
AI_Background_Remover(
        image_folder="Image_with_background", 
        image_name="R.jpg",
        result_image_folder="Image_without_background",
        result_image_name=f"R_rm_bg.png"
        )

# To Remove all images background that are save on "Image_with_background" Folder
# for image_name in os.listdir("Image_with_background"):
#     AI_Background_Remover(
#         image_folder="Image_with_background", 
#         image_name=image_name,
#         result_image_folder="Image_without_background",
#         result_image_name=f"{image_name[:-4]}_rm_bg.png"
#         )
#     print(f"{image_name} Background Removed")
