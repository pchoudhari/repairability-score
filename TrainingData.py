import argparse

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json
import re
from youtube_transcript_api import YouTubeTranscriptApi
from deepmultilingualpunctuation import PunctuationModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"
repairabilityscore_question  = "What is the repairability score of this phone?"
# repairabilityscore_question  = "Give a repairability score of this phone on a scale of 1 to 10 based on the above teardown, and also summarize it's main points."

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
DEVELOPER_KEY = 'AIzaSyDxPFaU4q0TQDWz7mM3ne6kKUKd7r3jKJE'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
YOUTUBE_QUERY = 'mobile teardown disassembly'
YOUTUBE_CHANNELID_LIST = [
                        #   ['UC20o1OUWGrixnCEE1gQ9clg',200],
                          ['UCHbx9IUW7eCeJsC4sBCTNBA',200]
                          ]
YOUTUBE_MAX_RESULTS = 50

punctuationModel = PunctuationModel()
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Get the youtube IDs for the channelIDs
def youtube_search(youtubeChannelId, nextPage):
    # Call the search.list method to retrieve results matching the specified
    # query term.
    
    # Check if the nextpagetoken has been sent
    if nextPage != '':
        search_response = youtube.search().list(
            q=YOUTUBE_QUERY,
            part='id,snippet',
            maxResults= YOUTUBE_MAX_RESULTS,
            channelId = youtubeChannelId,
            type = 'video',
            pageToken = nextPage      
        ).execute()
    else:
        search_response = youtube.search().list(
            q=YOUTUBE_QUERY,
            part='id,snippet',
            maxResults= YOUTUBE_MAX_RESULTS,
            channelId = youtubeChannelId,
            type = 'video',
            # nextPageToken = nextPage      
        ).execute()

    videos = []
    # channels = []
    # playlists = []
    videoIDs = []

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append('%s (%s)' % (search_result['snippet']['title'],
                                        search_result['id']['videoId']))
            videoIDs.append(search_result['id']['videoId'])
        # elif search_result['id']['kind'] == 'youtube#channel':
        #     channels.append('%s (%s)' % (search_result['snippet']['title'],
        #                                 search_result['id']['channelId']))
        # elif search_result['id']['kind'] == 'youtube#playlist':
        #     playlists.append('%s (%s)' % (search_result['snippet']['title'],
        #                                 search_result['id']['playlistId']))

    print('Videos:\n', '\n'.join(videos), '\n')
#   print('Channels:\n', '\n'.join(channels), '\n')
#   print('Playlists:\n', '\n'.join(playlists), '\n')

    return videoIDs, search_response['nextPageToken']


# Return the transcript for the Youtube ID
def GetTranscript(youtubeID):
    
    # retrieve the available transcripts
    transcript_list = YouTubeTranscriptApi.list_transcripts(youtubeID)
    
    retString = ""
    
    # iterate over all available transcripts
    for transcript in transcript_list:  
        # fetch the actual transcript data
        transcript = transcript.fetch()
        # print(transcript)
        
        for index in range(len(transcript)):
            # print(transcript[index])
            tmp = (re.sub("([a-zA-Z0-9]+)\'([a-zA-Z0-9]+)", "\\1\\2", str(transcript[index])).replace('\'', '\"')).replace('\\n', ' ').replace('\\xa0', ' ').replace(';', ' ')
            # retString += json.loads((re.sub("([a-zA-Z0-9]+)\'([a-zA-Z0-9]+)", "\\1\\2", str(transcript[index])).replace('\'', '\"')).replace('\n', ' '))['text']
            # print(tmp)
            retString += json.loads(tmp)['text']
            retString += " "
        
    # Punctuate the model
    result = punctuationModel.restore_punctuation(retString)

    return result

# Get the repairability score from the QnA Model
def getAnswer(question, context):
    
    QA_input = {
        'question': question,
        'context': context
        }
    res = nlp(QA_input)
    
    # print(str(res))
    
    answer = json.loads(re.sub("([a-zA-Z0-9]+)\'([a-zA-Z0-9]+)", "\\1\\2", str(res)).replace('\'', '\"'))['answer']
    score = json.loads(re.sub("([a-zA-Z0-9]+)\'([a-zA-Z0-9]+)", "\\1\\2", str(res)).replace('\'', '\"'))['score']
    
    # print(answer)
    
    return answer, round(score,3)

# Remove the sentence containing the repairability score from the transcript
def removeScoreFromTranscript(transcript, repairabilityScore):
    
    sentences = transcript.split('.')
    
    retString = ''
    
    for sentence in sentences:
        if repairabilityScore in sentence:
            continue
        
        retString += sentence + '.'
        
    return retString

if __name__ == '__main__':
    try:
        trainingDataFile = open("TraningData.tsv", "a")
        
        # Iterate through the channel list
        for channel in YOUTUBE_CHANNELID_LIST:
            youtubeChannelId = channel[0]
            youtubeChannelResults = channel[1]
            
            searchIterations = youtubeChannelResults//YOUTUBE_MAX_RESULTS
            
            # Repeatedly search since only 50 results are returned at a time
            nexPageToken=''
            for interation in range(0,searchIterations):
            
                nxtpageTkn = nexPageToken
                print('Next page token : ' + nxtpageTkn)
                youtubeIDList,  nexPageToken= youtube_search(youtubeChannelId, nxtpageTkn)
                # youtubeIDList = ["EHRoo1QffL0"]

                for youtubeID in youtubeIDList:
                    try:
                        # Get the transcript for the youtube ID
                        transcript = GetTranscript(youtubeID)
                        
                        # Get the repairability score from QnA model
                        repairabilityScore, score = getAnswer(repairabilityscore_question, transcript)
                        
                        # Add to training data only if score > 0.5
                        if score > 0.5:                
                            # Remove the sentence containing the repairability score
                            cleanTranscript = removeScoreFromTranscript(transcript, repairabilityScore)
                            
                            # Write to training data file
                            trainingDataFile.write("context:"+cleanTranscript+"question:"+repairabilityscore_question+"\t"+repairabilityScore+"\n")
                                    
                            print(youtubeID + "\t" + repairabilityScore)  
                    except:
                        # Continue with next youtubeid if error in retrieving transcript
                        continue
           
        trainingDataFile.close()
                
    except HttpError as e:
        print('An HTTP error %d occurred:\n%s') % (e.resp.status, e.content)
    # print('An HTTP error %d occurred:\n%s')