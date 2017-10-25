import httplib2
import json
from time import sleep

def ScrapeStats(name):
    conn = httplib2.Http()
    resp, content = conn.request(uri = 'https://owapi.net/api/v3/u/{}/blob'.format(name))
    if content:
        return json.loads(content.decode('utf-8'))
    else:
        return {"msg": "content empty", "error": 499}

def DetermineRegion(data):
    regions = ['eu','kr','us']
    rank=0
    player_region = None
    for region in regions:
        if data[region]:
            if data[region]['stats']['competitive']:
                if data[region]['stats']['competitive']['overall_stats']['comprank']:
                    if data[region]['stats']['competitive']['overall_stats']['comprank']>rank:
                        player_region = region
    return player_region


with open('nameslist3.txt','r') as f:
       content = f.readlines()
       for line in content:
           names = line.split(",")

def GetStatsFromRegion(data, region):
    if not region:
        return
    else:
        output = data[region]
        output['stats']['competitive'].pop('competitive', None)
        output['stats'].pop('quickplay', None)
        output['heroes']['stats'].pop('quickplay', None)
        output['heroes']['playtime'].pop('quickplay', None)
        output.pop('achievements', None)
        return output


i=281
# data = []
while i < 7954:
    name = names[i]
    new_data = ScrapeStats(name)
    if 'msg' not in new_data:
        region = DetermineRegion(new_data)
        new_data = GetStatsFromRegion(new_data, region)
        if new_data:
            with open("playerstats2.txt", 'a') as f:
                f.write(str(new_data) + ","+"\n")
        # data.append(new_data)
        # print(len(names))
        print(i)
        i+=1
    else:
        if new_data['msg'] == 'you are being ratelimited':
            sleep(8)
            print('ratelimited')
        else:
            print('profile {} does not exist'.format(i))
            i+=1

    sleep(5)
