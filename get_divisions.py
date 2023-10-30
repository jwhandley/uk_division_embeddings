import requests
import asyncio
import aiohttp
import json
import argparse
import logging
from tqdm.auto import tqdm
from tqdm.asyncio import tqdm as atqdm
import random
MAX_RETRIES = 5  # Maximum number of retries before giving up

semaphore = asyncio.Semaphore(5)

def get_num_divisions(start_date, end_date):
    url = f"https://commonsvotes-api.parliament.uk/data/divisions.json/searchTotalResults?queryParameters.startDate={start_date}&queryParameters.endDate={end_date}"
    r = requests.get(url)
    r.raise_for_status()

    return r.json()


def search_divisions(start_date, end_date):
    base_url = f"https://commonsvotes-api.parliament.uk/data/divisions.json/search?queryParameters.startDate={start_date}&queryParameters.endDate={end_date}"
    # First, find the total number of divisions
    num_divisions = get_num_divisions(start_date, end_date)

    # Number of requests = ceil(number of divisions / 25)
    num_requests = num_divisions // 25 + 1

    results = []
    for i in tqdm(range(num_requests), desc="Getting divisions"):
        r = requests.get(base_url + f"&queryParameters.skip={25*i}")
        r.raise_for_status()
        data = r.json()
        results.extend([row['DivisionId'] for row in data])


    return results

async def get_devision_details(division_id, session, semaphore):
    base_url = f"https://commonsvotes-api.parliament.uk/data/division/{division_id}.json"
    retries = 0
    while retries < MAX_RETRIES:
        async with semaphore:
            async with session.get(base_url) as r:
                if r.status == 429:  # Too Many Requests
                    # Wait for 2^retries + random milliseconds
                    sleep_time = (2 ** retries) + (random.randint(0, 1000) / 1000)
                    logging.warning(f"Too many requests. Waiting for {sleep_time} seconds")
                    retries += 1
                    await asyncio.sleep(sleep_time)
                    continue
                
                r.raise_for_status()
                return await r.json()
    raise Exception(f"Maximum retries reached for division_id: {division_id}")

async def get_divisions(division_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [get_devision_details(division_id, session, semaphore) for division_id in division_ids]
        return await atqdm.gather(*tasks, desc='Getting division details')

def process_division(division):
    votes = []
    for aye in division['Ayes']:
        # Filter out Northern Ireland parties
        if aye['PartyAbbreviation'] in ['Alliance','DUP','SDLP','Sinn Fein','UUP','Ind','Reclaim']:
            continue
        votes.append((division['DivisionId'],aye['MemberId'],1))

    for no in division['Noes']:
        # Filter out Northern Ireland parties
        if no['PartyAbbreviation'] in ['Alliance','DUP','SDLP','Sinn Fein','UUP','Ind','Reclaim']:
            continue
        votes.append((division['DivisionId'],no['MemberId'],0))

    return votes

async def get_member_details(member_id, session, semaphore):
    base_url = f"https://members-api.parliament.uk/api/Members/{member_id}"
    retries = 0
    while retries < MAX_RETRIES:
        async with semaphore:
            async with session.get(base_url) as r:
                if r.status == 429:
                    sleep_time = (2 ** retries) + (random.randint(0, 1000) / 1000)
                    logging.warning(f"Too many requests. Waiting for {sleep_time} seconds")
                    retries += 1
                    await asyncio.sleep(sleep_time)
                    continue

                r.raise_for_status()
                return await r.json()
    raise Exception(f"Maximum retries reached for member_id: {member_id}")

async def get_members(member_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [get_member_details(member_id, session, semaphore) for member_id in member_ids]
        return await atqdm.gather(*tasks, desc='Getting member details')

def process_member(member):
    return {"MemberId": member['value']['id'], "Name": member['value']['nameDisplayAs'], "PartyAbbrev": member['value']['latestParty']['abbreviation'], "Color": member['value']['latestParty']['backgroundColour']}

def save_results(path, results):
    with open(path, 'w') as f:
        json.dump(results, f)


async def main_async(start_date, end_date, output_path):
    division_ids = search_divisions(start_date, end_date)
    division_results = await get_divisions(division_ids)

    votes = []
    for division in tqdm(division_results, desc="Processing divisions"):
        votes.extend(process_division(division))

    division_id_map = {}
    member_id_map = {}
    new_division_id = 0
    new_member_id = 0

    for vote in votes:
        division_id, member_id, _ = vote

        if division_id not in division_id_map:
            division_id_map[division_id] = new_division_id
            new_division_id += 1

        if member_id not in member_id_map:
            member_id_map[member_id] = new_member_id
            new_member_id += 1

    for idx, vote in enumerate(votes):
        votes[idx] = (division_id_map[vote[0]], member_id_map[vote[1]], vote[2])

    member_ids = list(member_id_map.keys())
    member_details = await get_members(member_ids)
    member_details = [process_member(member) for member in member_details]

    save_results("output/member_details.json", member_details)
    save_results("output/member_id_map.json", member_id_map)
    save_results(output_path, votes)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', type=str, help='Start date in format YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, help='End date in format YYYY-MM-DD')
    parser.add_argument('--output', type=str, help='Output file path')
    args = parser.parse_args()

    asyncio.run(main_async(args.start_date, args.end_date, args.output))

if __name__ == '__main__':
    main()
