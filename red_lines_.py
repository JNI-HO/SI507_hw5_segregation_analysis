import json
import os
import random
from statistics import mean
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Polygon
import requests        
import re
from collections import Counter

random.seed(17)

GRADE_COLOR = {
    'A': 'darkgreen',
    'B': 'cornflowerblue',
    'C': 'gold',
    'D': 'maroon'
}


class DetroitDistrict:
    """
    A class representing a district in Detroit with attributes related to historical redlining.
    coordinates,holcGrade,holcColor,id,description should be load from the redLine data file
    if cache is not available

    Parameters 
    ------------------------------
    coordinates : list of lists, 2D List, not list of list of list
        The geographic coordinates (longitude, latitude pairs) that define the district's polygon.

    holcGrade : str
        HOLC grade for the district (e.g., 'A', 'B', 'C', 'D').

    holcColor : str, optional
        A color representing the HOLC grade (default is automatically derived).

    id : str
        Unique identifier for the district.

    description : str
        Qualitative description of the district.

    randomLat : float, optional
        Randomly selected latitude within the district boundary (default is None).

    randomLong : float, optional
        Randomly selected longitude within the district boundary (default is None).

    medIncome : float, optional
        Median household income for the district, to be filled later (default is None).
        
    censusTract : str, optional
        Census tract code for the district (default is None).


    Attributes
    ------------------------------
    self.coordinates 
    self.holcGrade 
    holcColor : str
        The color representation of the HOLC grade.
        • Districts with holc grade A should be assigned the color 'darkgreen'
        • Districts with holc grade B should be assigned the color 'cornflowerblue'
        • Districts with holc grade C should be assigned the color 'gold'
        • Districts with holc grade D should be assigned the color 'maroon'

    self.id 
    self.description 
    self.randomLat 
    self.randomLong 
    self.medIncome 
    self.censusTract 
    """

    def __init__(self,
                 coordinates: List[List[float]],
                 holcGrade: str,
                 id: str,
                 description: str,
                 holcColor: Optional[str] = None,
                 randomLat: Optional[float] = None,
                 randomLong: Optional[float] = None,
                 medIncome: Optional[float] = None,
                 censusTract: Optional[str] = None):
        self.coordinates = coordinates
        self.holcGrade = holcGrade
        self.holcColor = holcColor if holcColor is not None else GRADE_COLOR.get(holcGrade, 'black')
        self.id = id
        self.description = description
        self.randomLat = randomLat
        self.randomLong = randomLong
        self.medIncome = medIncome
        self.censusTract = censusTract

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "holcGrade": self.holcGrade,
            "holcColor": self.holcColor,
            "id": self.id,
            "description": self.description,
            "randomLat": self.randomLat,
            "randomLong": self.randomLong,
            "medIncome": self.medIncome,
            "censusTract": self.censusTract,
        }

    def __repr__(self) -> str:
        return f"DetroitDistrict(id={self.id}, grade={self.holcGrade})"


class RedLines:
    """
    A class to handle a collection of DetroitDistrict objects, providing functionality to create districts from
    data, plot them, generate random points within each district, fetch census tract information, retrieve income
    data, cache and load data, and compute various statistics.
    """

    def __init__(self, cacheFile: Optional[str] = None):
        """
        Initializes the RedLines object with districts loaded from a cache file if provided.

        Parameters
        ----------
        cacheFile : str, optional
            Path to a JSON cache file containing district data. If provided, the method attempts to load
            districts from the cache file. If loading fails or the file does not exist, `self.districts` is set 
            to an empty list.

        Note
        ----------
        You should revise your init method to enable reading district data from a cache file.
        """
        self.districts: List[DetroitDistrict] = []
        if cacheFile and self.loadCache(cacheFile):
            return
        self.districts = []

    def createDistricts(self, fileName: str):
        """
        Creates DetroitDistrict instances from redlining data in a specified file.
        Based on the understanding in step 1, load the file,parse the json object, 
        and create 238 districts instance.
        Finally, store districts instance in a list, 
        and assign the list to be districts attribute of RedLines.

        Parameters
        ----------
        fileName : str
            The name of the file containing redlining data in JSON format.

        Hint
        ----------
        The data for description attribute could be from  
        one of the dict key with only number.
        """
        if self.districts:
            return
        with open(fileName, "r", encoding="utf-8") as f:
            data = json.load(f)

        districts: List[DetroitDistrict] = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})

            # Coordinates handling: MultiPolygon -> take outer ring; Polygon -> ring
            coords = geom.get("coordinates", [])
            coordinates = []
            if geom.get("type") == "MultiPolygon":
                if coords and isinstance(coords[0], list) and isinstance(coords[0][0], list):
                    coordinates = coords[0][0]
            elif geom.get("type") == "Polygon":
                if coords and isinstance(coords[0], list):
                    coordinates = coords[0]
            else:
                # fallback best-effort
                if coords and isinstance(coords[0], list):
                    if coords and isinstance(coords[0][0], list):
                        coordinates = coords[0]
                    else:
                        coordinates = coords

            holcGrade = props.get("holc_grade") or ""
            holc_id = str(props.get("holc_id") or "")

            # description：取 area_description_data 字典中最大數字鍵對應的文字
            desc = ""
            area_desc = props.get("area_description_data")
            if isinstance(area_desc, dict):
                numeric_items = [(int(k), v) for k, v in area_desc.items() if str(k).isdigit()]
                if numeric_items:
                    numeric_items.sort(key=lambda kv: kv[0])
                    desc = str(numeric_items[-1][1]).strip()
            elif isinstance(area_desc, str):
                desc = area_desc.strip()

            districts.append(
                DetroitDistrict(
                    coordinates=coordinates,
                    holcGrade=holcGrade,
                    id=holc_id,
                    description=desc,
                    holcColor=GRADE_COLOR.get(holcGrade, "black"),
                )
            )
        self.districts = districts

    def plotDistricts(self):
        """ 
        Plots the districts using matplotlib, displaying each  
        district's location and color. 
        Name it redlines_graph.png and save it to the current  
        directory. 
        """ 
        fig, ax = plt.subplots()
        for d in self.districts:
            if not d.coordinates:
                continue
            poly = Polygon(d.coordinates, closed=True, edgecolor='black', facecolor=d.holcColor, linewidth=0.5)
            ax.add_patch(poly)
        ax.autoscale()
        plt.rcParams["figure.figsize"] = (15, 15)
        plt.axis('equal')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Detroit HOLC Redlining Map")
        plt.savefig("redlines_graph.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
        return True

    def generateRandPoint(self):
        """
        Generates a random (lon, lat) point inside each district polygon using a fixed grid
        and Path.contains_points. Stores the chosen point in each district's randomLong/randomLat.
        Returns True if executed, False otherwise.
        """
        if not self.districts:
            return False

        xgrid = np.arange(-83.5, -82.8, .004)
        ygrid = np.arange(42.1, 42.6, .004)
        xmesh, ymesh = np.meshgrid(xgrid, ygrid)
        points = np.vstack((xmesh.flatten(), ymesh.flatten())).T  # (N,2)

        for j in self.districts:
            if not j.coordinates:
                continue
            p = Path(j.coordinates)
            mask = p.contains_points(points)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                xs = [pt[0] for pt in j.coordinates]
                ys = [pt[1] for pt in j.coordinates]
                j.randomLong = float(sum(xs) / len(xs))
                j.randomLat = float(sum(ys) / len(ys))
            else:
                lon, lat = points[random.choice(idxs)]
                j.randomLong = float(lon)
                j.randomLat = float(lat)

        return True

    def fetchCensus(self):
        """
        For each district, use its randomLat/randomLong to call the FCC Census API and store a
        9-digit census tract FIPS (county(3)+tract(6)) as district.censusTract.
        The API requires the 'censusYear' aligned with the ACS year we'll use in Step 7.
        """
        if not self.districts:
            return
        for d in self.districts:
            if d.randomLat is None or d.randomLong is None:
                continue
            url = f"https://geo.fcc.gov/api/census/area?lat={d.randomLat}&lon={d.randomLong}&censusYear=2010&format=json"
            try:
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                data = r.json()
                block_fips = data['results'][0]['block_fips']  # 15 digits: SS + CCC + TTTTTT + BBBB
                d.censusTract = block_fips[2:11]                # CCC + TTTTTT = 9 digits
            except Exception:
                d.censusTract = None

    def fetchIncome(self, api_key: Optional[str] = None):
        """
        Retrieves the median household income (B19013_001E) for each district based on the census tract.
        One API call per (state, county). Map by the last 6 digits of the tract code.
        Assumes Michigan (state=26) for Detroit data.
        """
        tracts = [d.censusTract for d in self.districts if d.censusTract]
        if not tracts:
            return

        STATE = "26"
        counties = sorted(set(t[:3] for t in tracts))  # first 3 of 9-digit = county

        base = "https://api.census.gov/data/2018/acs/acs5"
        var = "B19013_001E"
        tract6_to_income: Dict[str, float] = {}

        for ccc in counties:
            params = {
                "get": f"NAME,{var}",
                "for": "tract:*",
                "in": f"state:{int(STATE)} county:{int(ccc)}",
            }
            if api_key:
                params["key"] = api_key
            try:
                r = requests.get(base, params=params, timeout=30)
                r.raise_for_status()
                rows = r.json()
                header = rows[0]
                var_idx = header.index(var)
                tract_idx = header.index("tract")
                for row in rows[1:]:
                    tract6 = row[tract_idx].zfill(6)
                    try:
                        income = float(row[var_idx])
                        if income < 0:
                            income = 0.0
                    except Exception:
                        income = 0.0
                    tract6_to_income[tract6] = income
            except Exception:
                continue

        for d in self.districts:
            if d.censusTract:
                tract6 = d.censusTract[-6:]
                d.medIncome = float(tract6_to_income.get(tract6, 0.0))

    def cacheData(self, fileName: str):
        """
        Create a JSON cache that cache all the information of each district instance.
        """
        data = [dist.to_dict() for dist in self.districts]
        with open(fileName, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def loadCache(self, fileName: str) -> bool:
        """
        Load district instances from cache file if available.
        Hint self.districts = [ DetroitDistrict(**data) for data in district_data]
        """
        if not os.path.exists(fileName):
            return False
        try:
            with open(fileName, "r", encoding="utf-8") as f:
                arr = json.load(f)
            self.districts = [DetroitDistrict(**d) for d in arr]
            return True
        except Exception:
            self.districts = []
            return False

    def calcIncomeStats(self):
        """
        Calculates the mean and median of median household incomes for each district grade (A, B, C, D).
        Returns a list: [AMean, AMedian, BMean, BMedian, CMean, CMedian, DMean, DMedian] (rounded to int).
        """
        out: List[int] = []
        for g in ['A', 'B', 'C', 'D']:
            vals = [float(d.medIncome) for d in self.districts if d.holcGrade == g and isinstance(d.medIncome, (int, float))]
            if vals:
                a_mean = int(round(float(np.mean(vals))))
                a_med = int(round(float(np.median(vals))))
                out.extend([a_mean, a_med])
            else:
                out.extend([0, 0])
        return out

    def findCommonWords(self):
        """
        Analyzes the qualitative descriptions of each district category (A, B, C, D) and identifies the
        10 most common words unique to each category.

        This method aggregates the qualitative descriptions for each district category, splits them into
        words, and computes the frequency of each word. It then identifies and returns the 10 most 
        common words that are unique to each category, excluding common English filler words.

        Returns
        -------
        list of lists
            A list containing four lists, each list containing the 10 most common words for each 
            district category (A, B, C, D). The first list should represent grade A, and second for grade B,etc.
            The words should be in the order of their frequency.

        Notes
        -----
        - Common English filler words such as 'the', 'of', 'and', etc., are excluded from the analysis.
        - The method ensures that the common words are unique across the categories, i.e., no word 
        appears in more than one category's top 10 list.
        - Regular expressions could be used for word splitting to accurately capture words
        """

        filler_words = set(['the', 'of', 'and', 'in', 'to', 'a', 'is', 'for', 'on', 'that'])

        buckets = {'A': [], 'B': [], 'C': [], 'D': []}
        for d in self.districts:
            if not d.description:
                continue
            text = " ".join(str(v) for v in d.description.values()) if isinstance(d.description, dict) else str(d.description)
            # tokenize
            words = re.findall(r"[A-Za-z']+", text.lower())
            words = [w for w in words if w not in filler_words and len(w) > 2]
            buckets[d.holcGrade].extend(words)

        counts = {g: Counter(ws) for g, ws in buckets.items()}

        # Remove words that appear in multiple grades to ensure uniqueness
        all_words = {}
        for g, c in counts.items():
            for w in c:
                all_words.setdefault(w, set()).add(g)

        unique_counts = {g: Counter() for g in "ABCD"}
        for w, grades in all_words.items():
            if len(grades) == 1:
                g = next(iter(grades))
                unique_counts[g][w] = counts[g][w]

        # Get top 10 per grade in A, B, C, D order
        result = []
        for g in "ABCD":
            top10 = [w for w, _n in unique_counts[g].most_common(10)]
            result.append(top10)
            
        return result


def main():
    rl = RedLines(cacheFile=None)
    if os.path.exists("redlines_data.json"):
        rl.createDistricts("redlines_data.json")
        rl.plotDistricts()
        rl.generateRandPoint()
        rl.fetchCensus()
        # rl.fetchIncome(api_key="YOUR_CENSUS_API_KEY")
        rl.cacheData("redlines_cache.json")
        _ = RedLines(cacheFile="redlines_cache.json")
        print("Income stats (list):", rl.calcIncomeStats())
        print("Common words:", rl.findCommonWords())
    else:
        print("Place 'redlines_data.json' in the current directory to run the demo.")


if __name__ == "__main__":
    main()
