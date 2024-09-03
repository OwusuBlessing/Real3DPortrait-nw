import subprocess
import json
def extract_fourth_video(input_video_path, output_video_path):
    # Get the dimensions of the combined video using ffprobe
    def get_video_dimensions(video_path):
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=width,height', '-of', 'json', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
        dimensions = json.loads(result.stdout)['streams'][0]
        return dimensions['width'], dimensions['height']

    combined_width, combined_height = get_video_dimensions(input_video_path)

    # Calculate the width of each individual video
    individual_width = combined_width // 5
    individual_height = combined_height

    # Ensure the crop width does not exceed the video width
    if 4 * individual_width >= combined_width:
        print("Invalid crop width calculated. Please check the input video dimensions.")
        return None

    # Use ffmpeg to crop the fifth video
    crop_command = [
        'ffmpeg', '-i', input_video_path,
        '-vf', f'crop={individual_width}:{individual_height}:{4*individual_width}:0',
        '-c:a', 'copy',  # This line copies the audio codec
        output_video_path
    ]

    subprocess.run(crop_command)

    # Return the path to the new video
    return output_video_path