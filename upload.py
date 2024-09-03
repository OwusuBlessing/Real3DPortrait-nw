from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
import secrets
import os 
from extractor import extract_fourth_video
pub =   "public_hb1iwGN/UWT+aYg9mMkUQNeFh40="
private  = "private_BXnyXGjdhBpBs/sU7avdyLwPx1o="
id = "6pxd8st0ugi"



imagekit = ImageKit(
    private_key=private,
    public_key=pub,
    url_endpoint=id
)

def upload_to_imgkit(path,user_id):
        hex_string = secrets.token_hex(16)
        file_name = f"{user_id}_{hex_string}.mp4"
        output_directory = "result"
        new_video_path = os.path.join(output_directory, file_name)
        #new_video_path = extract_fourth_video(path, new_video_path)
        
        with open(path, 'rb') as file:
                    upload = imagekit.upload_file(
                        file=file,
                        file_name=file_name,
                        options=UploadFileRequestOptions(
                            response_fields=["is_private_file", "tags"],
                            tags=["tag1", "tag2"]
                        )
                    )
       
        result = upload.response_metadata.raw["url"]
        os.remove(path)
        #os.remove(new_video_path)
        print(f"Removed first video : {path} and 4th extractor {new_video_path}")
        return result


