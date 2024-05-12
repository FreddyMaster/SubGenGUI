from flask import Flask, request, render_template, send_file
from flaskwebgui import FlaskUI
from faster_whisper import WhisperModel
import tempfile
import os

app = Flask(__name__)
ui = FlaskUI(app)

# Function to format seconds into a time string in the format hh:mm:ss,ms
def format_time(seconds):
    """
    Formats a time duration in seconds into a string in the format HH:MM:SS,mmm.

    Args:
        seconds (int): The time duration in seconds.

    Returns:
        str: The formatted time string.

    Example:
        >>> format_time(3661)
        '01:01:01,000'
    """
    # Calculate minutes and seconds
    minutes, seconds = divmod(seconds, 60)
    # Calculate hours and minutes
    hours, minutes = divmod(minutes, 60)
    # Calculate milliseconds
    milliseconds = (seconds - int(seconds)) * 1000
    # Return formatted time string
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

# Function to transcribe audio from a video file
def transcribe_video(file_stream, model_size="medium", device="cpu"):
    """
    Transcribes a video file using the WhisperModel.

    Args:
        file_stream (file): The video file to transcribe.
        model_size (str, optional): The size of the WhisperModel to use. Defaults to "medium".
        device (str, optional): The device to run the WhisperModel on. Defaults to "cpu".

    Returns:
        tuple: A tuple containing the transcribed segments and information about the transcription.

    Raises:
        Exception: If an error occurs during the transcription process.
    """
    try:
        # Initialize the WhisperModel with specified model size and device
        model = WhisperModel(model_size, device=device, cpu_threads=8, compute_type="int8")
        
        # Create a temporary file to hold the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(file_stream.read())
            tmp_file_path = tmp_file.name
        
        # Transcribe the temporary file
        segments, info = model.transcribe(tmp_file_path, beam_size=5, vad_filter=True)
        
        # Clean up the temporary file
        os.remove(tmp_file_path)
        
        # Print detected language and its probability
        print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
        return segments, info
    except Exception as e:
        print(f"Error transcribing video: {e}")
        return None, None
    
# Function to write transcription segments to an SRT file
def write_srt_file(segments, info, filename):
    """
    Writes segments to an SRT file.

    Args:
        segments (list): A list of Segment objects.
        info (dict): A dictionary containing information about the segments.
        filename (str): The name of the input file.

    Returns:
        None

    Raises:
        IOError: If there is an error writing to the SRT file.
    """
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create the output directory if it doesn't exist

    # Generate SRT filename from input file name
    srt_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.srt')
    
    try:
        # Open the SRT file for writing
        with open(srt_filename, 'w', encoding='utf-8') as srt_file:
            for segment in segments:
                # Format start and end times
                start_time = format_time(segment.start)
                end_time = format_time(segment.end)

                # Extract text and segment ID
                text = segment.text
                segment_id = segment.id + 1

                # Format SRT line
                line_out = f"{segment_id}\n{start_time} --> {end_time}\n{text.lstrip()}\n\n"

                # Print and write SRT line
                print(line_out)
                srt_file.write(line_out)
    except IOError as e:
        print(f"Error writing to SRT file: {e}")
        
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    This function handles the index page of the web application.
    
    Returns:
        If the request method is 'POST':
            - If a file is uploaded, it calls the transcribe_video function with the file stream and other parameters.
            - If transcription is successful, it generates an SRT file and returns a success message with the file path.
            - If transcription fails, it returns a failure message.
            - If no file is uploaded, it returns a message indicating that no file was uploaded.
        If the request method is not 'POST', it renders the index.html template.
    """
    if request.method == 'POST':
        uploaded_file = request.files['file']
        model_size = request.form.get('model_size', 'medium')
        device = request.form.get('device', 'cpu')
        
        if uploaded_file and uploaded_file.filename:
            # Call your transcription function with the file stream
            segments, info = transcribe_video(uploaded_file.stream, model_size, device)
            
            if segments and info:
                # Generate SRT file in the /output directory
                write_srt_file(segments, info, uploaded_file.filename)
                srt_filename = os.path.splitext(uploaded_file.filename)[0] + '.srt'
                # Construct the full path to the SRT file
                srt_file_path = os.path.join('output', srt_filename)
                # Return the SRT file for download
                return send_file(srt_file_path, as_attachment=True)
            else:
                return "Failed to transcribe video."
        else:
            return "No file uploaded."
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True)
    # FlaskUI(app=app, server="flask").run()