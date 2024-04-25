from django.shortcuts import render
import requests
import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
GEMINI_PROMPT = " question:Give a repairability score of this phone on a scale of 1 to 10 based on the above teardown, and also summarize it's main points."
import markdown
TRANSCRIPT_ERROR = "Error getting the transcript for this video. Check if the video has transcript"
import os.path


def homepage(request):
    # Check if the youtube id is available
    youtubeID = request.POST.get('youtubeid', "")
    
    # Check if the transcript is available   
    youtubeTranscript = request.POST.get('transcript', "")

    # If youtubeID entered generate the youtube transcipt button pressed
    getTranscriptButton = request.POST.get('getTranscriptButton', False)
    try:
        if youtubeID != "" and getTranscriptButton == "True":
            # retrieve the available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(youtubeID)
            
            # iterate over all available transcripts
            for transcript in transcript_list:

                # fetch the actual transcript data
                transcript = transcript.fetch()
                
                for index in range(len(transcript)):
                    tmp = (re.sub("([a-zA-Z0-9]+)\'([a-zA-Z0-9]+)", "\\1\\2", str(transcript[index])).replace('\'', '\"')).replace('\\n', ' ').replace('\\xa0', ' ').replace(';', ' ')

                    youtubeTranscript += json.loads(tmp)['text']
                    youtubeTranscript += " "
    except:
        youtubeTranscript = TRANSCRIPT_ERROR
    
    repairabilitySummary = ''
    
    # Re-Generate the summary only if Get Repairability button is pressed
    repairabilityButton = request.POST.get('repairabilityButton', False)
    if youtubeTranscript != '' and repairabilityButton == "True":
        # Call gemini to get the repairability score
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])

        model = genai.GenerativeModel(model_name=f'models/gemini-pro')
        
        response = model.generate_content(youtubeTranscript + GEMINI_PROMPT)
        
        repairabilitySummary = response.text
     
    return render(
        request,
        'homepage.html',        
            {
                'youtubeID': youtubeID,
                'youtubetranscript': youtubeTranscript,
                'repairabilitySummary': markdown.markdown(repairabilitySummary.replace(" *","\n\n*")),
            }
            
        )

def aboutpage(request):
    return render(request, 'aboutpage.html', context={})
