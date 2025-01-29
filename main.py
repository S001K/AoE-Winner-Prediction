"""
execute in this format to access model result more clearly (python3 or python)
"python3 foo.py > model_output.txt"
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, make_scorer,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.svm import SVC

raw_csv = 'snapshots_arabia_t_4920.csv'
input_csv = "4920.csv"
data = pd.read_csv(raw_csv)
df = pd.DataFrame(data)


#Irrevalent Columns------------------------------------------------
df = df.drop('match_id', axis=1)
df = df.drop('time', axis=1)
df = df.drop('map', axis=1)
df = df.drop('map_size', axis=1)
df = df.drop('duration', axis=1)
df = df.drop('p1 Imperial Age Time', axis=1)
df = df.drop("p2 Imperial Age Time", axis=1)

#Merging 2 columns--------------------------------------------
df['civ_matchup'] = df['p1_civ'] + '/' + df['p2_civ']
df = df.drop('p1_civ', axis=1)
df = df.drop('p2_civ', axis=1)
col_matchup = df.pop('civ_matchup')
df.insert(7, "civ_matchup", col_matchup)


print("Merging Building Columns...")
#Merging Building Columns -----------------------------
economy_columns = ['p1_Town Center', 'p1_Mill', 'p1_Farm', 'p1_Market',
'p1_Lumber Camp', 'p1_Mining Camp', 'p1_Feitoria', 'p1_Fish Trap', 'p1_Fishing Ship', 'p1_House', 'p1_Rice Farm', 'p1_Wonder']

research_columns = ['p1_Blacksmith', 'p1_University']

mili_prod_columns = ['p1_Barracks', 'p1_Archery Range', 'p1_Stable', 'p1_Siege Workshop', 'p1_Dock',
'p1_Monastery']

defensive_columns = ['p1_Palisade Gate', 'p1_Krepost', 'p1_Gate', 'p1_Castle', 'p1_Outpost',
'p1_Watch Tower', 'p1_Donjon', 'p1_Bombard Tower']


df['p1_building_economic'] = df[economy_columns].sum(axis=1)
df['p1_building_research'] = df[research_columns].sum(axis=1)
df['p1_building_military'] = df[mili_prod_columns].sum(axis=1)
df['p1_building_defensive'] = df[defensive_columns].sum(axis=1)

df.drop(economy_columns, axis=1, inplace=True)
df.drop(research_columns, axis=1, inplace=True)
df.drop(mili_prod_columns, axis=1, inplace=True)
df.drop(defensive_columns, axis=1, inplace=True)

col_matchup = df.pop('p1_building_economic')
df.insert(7, "p1_building_economic", col_matchup)

col_matchup = df.pop('p1_building_research')
df.insert(8, "p1_building_research", col_matchup)

col_matchup = df.pop('p1_building_military')
df.insert(9, "p1_building_military", col_matchup)

col_matchup = df.pop('p1_building_defensive')
df.insert(10, "p1_building_defensive", col_matchup)

#Merging Building P2 Columns-------------------------------------------------------------------------------
economy_columns = ['p2_Town Center', 'p2_Mill', 'p2_Farm', 'p2_Market',
'p2_Lumber Camp', 'p2_Mining Camp', 'p2_Feitoria', 'p2_Fish Trap', 'p2_Fishing Ship', 'p2_House', 'p2_Rice Farm', 'p2_Wonder']

research_columns = ['p2_Blacksmith', 'p2_University']

mili_prod_columns = ['p2_Barracks', 'p2_Archery Range', 'p2_Stable', 'p2_Siege Workshop', 'p2_Dock',
'p2_Monastery']

defensive_columns = ['p2_Palisade Gate', 'p2_Krepost', 'p2_Gate', 'p2_Castle', 'p2_Outpost',
'p2_Watch Tower', 'p2_Donjon', 'p2_Bombard Tower']


df['p2_building_economic'] = df[economy_columns].sum(axis=1)
df['p2_building_research'] = df[research_columns].sum(axis=1)
df['p2_building_military'] = df[mili_prod_columns].sum(axis=1)
df['p2_building_defensive'] = df[defensive_columns].sum(axis=1)

df.drop(economy_columns, axis=1, inplace=True)
df.drop(research_columns, axis=1, inplace=True)
df.drop(mili_prod_columns, axis=1, inplace=True)
df.drop(defensive_columns, axis=1, inplace=True)

col_matchup = df.pop('p2_building_economic')
df.insert(7, "p2_building_economic", col_matchup)

col_matchup = df.pop('p2_building_research')
df.insert(8, "p2_building_research", col_matchup)

col_matchup = df.pop('p2_building_military')
df.insert(9, "p2_building_military", col_matchup)

col_matchup = df.pop('p2_building_defensive')
df.insert(10, "p2_building_defensive", col_matchup)


# Merging Unit Columnn--------------------------------------------
print("Merging Unit Columns...")

unit_ship = ['p1_Fire Galley', 'p1_Demolition Raft', 'p1_Galley', 'p1_Transport Ship', 'p1_Longboat', 'p1_Cannon Galleon', 'p1_Turtle Ship',
'p1_Caravel']

unit_siege = ['p1_Battering Ram', 'p1_Trebuchet', 'p1_Mangonel', 'p1_Scorpion', 'p1_Petard', 'p1_Bombard Cannon', 'p1_Organ Gun', 
'p1_Siege Tower', 'p1_Flaming Camel']

unit_infantry = ['p1_Spearman', 'p1_Militia', 'p1_Konnik', 'p1_Huskarl', 'p1_Shotel Warrior', 'p1_Gbeto', 'p1_Throwing Axeman', 
'p1_Eagle Scout', 'p1_Teutonic Knight', 'p1_Berserk', 'p1_Woad Raider', 'p1_Samurai', 'p1_Jaguar Warrior', 'p1_Obuch', 'p1_Serjeant', 
'p1_Kamayuk', 'p1_Karambit Warrior', 'p1_Condottiero', 'p1_Flemish Militia', 'p1_Xolotl Warrior']

unit_cavalry = ['p1_Knight', 'p1_Scout Cavalry', 'p1_Magyar Huszar', 'p1_Boyar', 'p1_Steppe Lancer', 'p1_Keshik', 'p1_Battle Elephant', 
'p1_Cataphract', 'p1_War Elephant', 'p1_Ballista Elephant', 'p1_Tarkan', 'p1_Leitis', 'p1_Mameluke', 'p1_Coustillier']

unit_civilian = ['p1_Villager', 'p1_Trade Cog', 'p1_Trade Cart']

unit_monk = ['p1_Monk', 'p1_Missionary', 'p1_None']

unit_archer = ['p1_Archer', 'p1_Skirmisher', 'p1_Camel Rider', 'p1_Kipchak', 'p1_Cavalry Archer', 'p1_Chu Ko Nu', 'p1_Arambai',
'p1_Mangudai', 'p1_Plumed Archer', 'p1_Longbowman', 'p1_Rattan Archer', 'p1_Camel Archer', 'p1_Genitour', 'p1_Genoese Crossbowman', 
'p1_Hussite Wagon', 'p1_War Wagon', 'p1_Slinger', 'p1_Elephant Archer', 'p1_Elite Kipchak', 'p1_Cavalier']

unit_gunpowder = ['p1_Conquistador', 'p1_Janissary', 'p1_Hand Cannoneer']

df['p1_unit_ship'] = df[unit_ship].sum(axis=1)
df['p1_unit_siege'] = df[unit_siege].sum(axis=1)
df['p1_unit_infantry'] = df[unit_infantry].sum(axis=1)
df['p1_unit_cavalry'] = df[unit_cavalry].sum(axis=1)
df['p1_unit_civilian'] = df[unit_civilian].sum(axis=1)
df['p1_unit_monk'] = df[unit_monk].sum(axis=1)
df['p1_unit_archer'] = df[unit_archer].sum(axis=1)
df['p1_unit_gunpowder'] = df[unit_gunpowder].sum(axis=1)

df.drop(unit_ship, axis=1, inplace=True)
df.drop(unit_siege, axis=1, inplace=True)
df.drop(unit_infantry, axis=1, inplace=True)
df.drop(unit_cavalry, axis=1, inplace=True)
df.drop(unit_civilian, axis=1, inplace=True)
df.drop(unit_monk, axis=1, inplace=True)
df.drop(unit_archer, axis=1, inplace=True)
df.drop(unit_gunpowder, axis=1, inplace=True)

col_matchup = df.pop('p1_unit_ship')
df.insert(10, "p1_unit_ship", col_matchup)

col_matchup = df.pop('p1_unit_siege')
df.insert(11, "p1_unit_siege", col_matchup)

col_matchup = df.pop('p1_unit_infantry')
df.insert(12, "p1_unit_infantry", col_matchup)

col_matchup = df.pop('p1_unit_cavalry')
df.insert(13, "p1_unit_cavalry", col_matchup)

col_matchup = df.pop('p1_unit_civilian')
df.insert(13, "p1_unit_civilian", col_matchup)

col_matchup = df.pop('p1_unit_monk')
df.insert(13, "p1_unit_monk", col_matchup)

col_matchup = df.pop('p1_unit_archer')
df.insert(13, "p1_unit_archer", col_matchup)

col_matchup = df.pop('p1_unit_gunpowder')
df.insert(13, "p1_unit_gunpowder", col_matchup)

# Merging P2 Unit Columnn--------------------------------------------
unit_ship = ['p2_Fire Galley', 'p2_Demolition Raft', 'p2_Galley', 'p2_Transport Ship', 'p2_Longboat', 'p2_Cannon Galleon', 'p2_Turtle Ship',
'p2_Caravel']

unit_siege = ['p2_Battering Ram', 'p2_Trebuchet', 'p2_Mangonel', 'p2_Scorpion', 'p2_Petard', 'p2_Bombard Cannon', 'p2_Organ Gun', 
'p2_Siege Tower', 'p2_Flaming Camel']

unit_infantry = ['p2_Spearman', 'p2_Militia', 'p2_Konnik', 'p2_Huskarl', 'p2_Shotel Warrior', 'p2_Gbeto', 'p2_Throwing Axeman', 
'p2_Eagle Scout', 'p2_Teutonic Knight', 'p2_Berserk', 'p2_Woad Raider', 'p2_Samurai', 'p2_Jaguar Warrior', 'p2_Obuch', 'p2_Serjeant', 
'p2_Kamayuk', 'p2_Karambit Warrior', 'p2_Condottiero', 'p2_Flemish Militia', 'p2_Xolotl Warrior']

unit_cavalry = ['p2_Knight', 'p2_Scout Cavalry', 'p2_Magyar Huszar', 'p2_Boyar', 'p2_Steppe Lancer', 'p2_Keshik', 'p2_Battle Elephant', 
'p2_Cataphract', 'p2_War Elephant', 'p2_Ballista Elephant', 'p2_Tarkan', 'p2_Leitis', 'p2_Mameluke', 'p2_Coustillier']

unit_civilian = ['p2_Villager', 'p2_Trade Cog', 'p2_Trade Cart']

unit_monk = ['p2_Monk', 'p2_Missionary', 'p2_None']

unit_archer = ['p2_Archer', 'p2_Skirmisher', 'p2_Camel Rider', 'p2_Kipchak', 'p2_Cavalry Archer', 'p2_Chu Ko Nu', 'p2_Arambai',
'p2_Mangudai', 'p2_Plumed Archer', 'p2_Longbowman', 'p2_Rattan Archer', 'p2_Camel Archer', 'p2_Genitour', 'p2_Genoese Crossbowman', 
'p2_Hussite Wagon', 'p2_War Wagon', 'p2_Slinger', 'p2_Elephant Archer', 'p2_Elite Kipchak', 'p2_Cavalier']

unit_gunpowder = ['p2_Conquistador', 'p2_Janissary', 'p2_Hand Cannoneer']

df['p2_unit_ship'] = df[unit_ship].sum(axis=1)
df['p2_unit_siege'] = df[unit_siege].sum(axis=1)
df['p2_unit_infantry'] = df[unit_infantry].sum(axis=1)
df['p2_unit_cavalry'] = df[unit_cavalry].sum(axis=1)
df['p2_unit_civilian'] = df[unit_civilian].sum(axis=1)
df['p2_unit_monk'] = df[unit_monk].sum(axis=1)
df['p2_unit_archer'] = df[unit_archer].sum(axis=1)
df['p2_unit_gunpowder'] = df[unit_gunpowder].sum(axis=1)

df.drop(unit_ship, axis=1, inplace=True)
df.drop(unit_siege, axis=1, inplace=True)
df.drop(unit_infantry, axis=1, inplace=True)
df.drop(unit_cavalry, axis=1, inplace=True)
df.drop(unit_civilian, axis=1, inplace=True)
df.drop(unit_monk, axis=1, inplace=True)
df.drop(unit_archer, axis=1, inplace=True)
df.drop(unit_gunpowder, axis=1, inplace=True)

col_matchup = df.pop('p2_unit_ship')
df.insert(22, "p2_unit_ship", col_matchup)

col_matchup = df.pop('p2_unit_siege')
df.insert(22, "p2_unit_siege", col_matchup)

col_matchup = df.pop('p2_unit_infantry')
df.insert(22, "p2_unit_infantry", col_matchup)

col_matchup = df.pop('p2_unit_cavalry')
df.insert(22, "p2_unit_cavalry", col_matchup)

col_matchup = df.pop('p2_unit_civilian')
df.insert(22, "p2_unit_civilian", col_matchup)

col_matchup = df.pop('p2_unit_monk')
df.insert(22, "p2_unit_monk", col_matchup)

col_matchup = df.pop('p2_unit_archer')
df.insert(22, "p2_unit_archer", col_matchup)

col_matchup = df.pop('p2_unit_gunpowder')
df.insert(22, "p2_unit_gunpowder", col_matchup)

#####################################################################
print("Merging Research Columns...")
archery_range_tree = [
    "p1_Crossbowman", "p1_Elite Skirmisher", "p1_Arbalester",
    "p1_Elite Elephant Archer",
    "p1_Elite Genitour",
    "p1_Thumb Ring", "p1_Parthian Tactics", "p1_Imperial Skirmisher"
]

barracks_tree = [
    "p1_Man-at-Arms", "p1_Long Swordsman", "p1_Two-Handed Swordsman", "p1_Champion",
    "p1_Pikeman", "p1_Halberdier",
    "p1_Eagle Warrior", "p1_Elite Eagle Warrior",
    "p1_Supplies", "p1_Squires", "p1_Arson"
]
stable_tree = [
    "p1_Bloodlines",
    "p1_Light Cavalry","p1_Husbandry",
    "p1_Hussar", "p1_Heavy Camel Rider", "p1_Elite Battle Elephant",
    "p1_Winged Hussar", "p1_Paladin", "p1_Imperial Camel Rider"
]

siege_workshop_tree = [
    "p1_Capped Ram", "p1_Onager", "p1_Heavy Scorpion",
    "p1_Siege Ram", "p1_Siege Onager", "p1_Houfnice"
]

blacksmith_tree = [
    "p1_Padded Archer Armor", "p1_Fletching", "p1_Forging", "p1_Scale Barding Armor", "p1_Scale Mail Armor",
    "p1_Leather Archer Armor", "p1_Bodkin Arrow", "p1_Iron Casting", "p1_Chain Barding Armor", "p1_Chain Mail Armor",
    "p1_Ring Archer Armor", "p1_Bracer", "p1_Blast Furnace", "p1_Plate Barding Armor", "p1_Plate Mail Armor"
]

dock_tree = [
    "p1_Gillnets","p1_War Galley",
    "p1_Fast Fire Ship", "p1_Heavy Demolition Ship", "p1_Galleon", "p1_Elite Turtle Ship", "p1_Elite Longboat", "p1_Elite Caravel",
    "p1_Elite Cannon Galleon","p1_Dry Dock","p1_Careening", "p1_Shipwright"
]
university_tree = [
    "p1_Masonry", "p1_Fortified Wall", "p1_Ballistics", "p1_Guard Tower", "p1_Heated Shot", "p1_Murder Holes", "p1_Treadmill Crane",
    "p1_Architecture", "p1_Chemistry", "p1_Siege Engineers", "p1_Keep", "p1_Arrowslits",
]
castle_tree = [
    "p1_Hoardings", "p1_Sappers", "p1_Conscription", "p1_Spies/Treason"
]

donjon_tree = [
    "p1_Elite Serjeant", "p1_Elite Konnik"
]

monastery_tree = [
    "p1_Redemption", "p1_Atonement", "p1_Herbal Medicine", "p1_Heresy", "p1_Sanctity", "p1_Fervor",
    "p1_Illumination", "p1_Block Printing", "p1_Faith", "p1_Theocracy"
]

town_center_tree = [
    "p1_Feudal Age", "p1_Loom",
    "p1_Town Watch", "p1_Castle Age", "p1_Wheelbarrow",
    "p1_Town Patrol", "p1_Imperial Age", "p1_Hand Cart"
]

market_tree = [
    "p1_Gold Mining", "p1_Stone Mining", "p1_Double-Bit Axe",
    "p1_Gold Shaft Mining", "p1_Stone Shaft Mining", "p1_Bow Saw", "p1_Coinage", "p1_Caravan", "p1_Heavy Plow",
    "p1_Two-Man Saw", "p1_Banking", "p1_Guilds", "p1_Crop Rotation",
    "p1_Horse Collar"
]


# Archery Range Tree
df['p1_archery_range_tree'] = df[archery_range_tree].sum(axis=1)
df.drop(archery_range_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_archery_range_tree')
df.insert(31, "p1_archery_range_tree", col_matchup)

# Barracks Tree
df['p1_barracks_tree'] = df[barracks_tree].sum(axis=1)
df.drop(barracks_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_barracks_tree')
df.insert(31, "p1_barracks_tree", col_matchup)

# Stable Tree
df['p1_stable_tree'] = df[stable_tree].sum(axis=1)
df.drop(stable_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_stable_tree')
df.insert(31, "p1_stable_tree", col_matchup)

# Siege Workshop Tree
df['p1_siege_workshop_tree'] = df[siege_workshop_tree].sum(axis=1)
df.drop(siege_workshop_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_siege_workshop_tree')
df.insert(31, "p1_siege_workshop_tree", col_matchup)

# Blacksmith Tree
df['p1_blacksmith_tree'] = df[blacksmith_tree].sum(axis=1)
df.drop(blacksmith_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_blacksmith_tree')
df.insert(31, "p1_blacksmith_tree", col_matchup)

# Dock Tree
df['p1_dock_tree'] = df[dock_tree].sum(axis=1)
df.drop(dock_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_dock_tree')
df.insert(31, "p1_dock_tree", col_matchup)

# University Tree
df['p1_university_tree'] = df[university_tree].sum(axis=1)
df.drop(university_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_university_tree')
df.insert(31, "p1_university_tree", col_matchup)

# Castle Tree
df['p1_castle_tree'] = df[castle_tree].sum(axis=1)
df.drop(castle_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_castle_tree')
df.insert(31, "p1_castle_tree", col_matchup)

# Donjon Tree
df['p1_donjon_tree'] = df[donjon_tree].sum(axis=1)
df.drop(donjon_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_donjon_tree')
df.insert(31, "p1_donjon_tree", col_matchup)

# Monastery Tree
df['p1_monastery_tree'] = df[monastery_tree].sum(axis=1)
df.drop(monastery_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_monastery_tree')
df.insert(31, "p1_monastery_tree", col_matchup)

# Town Center Tree
df['p1_town_center_tree'] = df[town_center_tree].sum(axis=1)
df.drop(town_center_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_town_center_tree')
df.insert(31, "p1_town_center_tree", col_matchup)

# Market Tree
df['p1_market_tree'] = df[market_tree].sum(axis=1)
df.drop(market_tree, axis=1, inplace=True)
col_matchup = df.pop('p1_market_tree')
df.insert(31, "p1_market_tree", col_matchup)

archery_range_tree = [
    "p2_Crossbowman", "p2_Elite Skirmisher", "p2_Arbalester",
    "p2_Elite Elephant Archer",
    "p2_Elite Genitour",
    "p2_Thumb Ring", "p2_Parthian Tactics", "p2_Imperial Skirmisher"
]

barracks_tree = [
    "p2_Man-at-Arms", "p2_Long Swordsman", "p2_Two-Handed Swordsman", "p2_Champion",
    "p2_Pikeman", "p2_Halberdier",
    "p2_Eagle Warrior", "p2_Elite Eagle Warrior",
    "p2_Supplies", "p2_Squires", "p2_Arson"
]
stable_tree = [
    "p2_Bloodlines",
    "p2_Light Cavalry","p2_Husbandry",
    "p2_Hussar", "p2_Heavy Camel Rider", "p2_Elite Battle Elephant",
    "p2_Winged Hussar", "p2_Paladin", "p2_Imperial Camel Rider"
]

siege_workshop_tree = [
    "p2_Capped Ram", "p2_Onager", "p2_Heavy Scorpion",
    "p2_Siege Ram", "p2_Siege Onager", "p2_Houfnice"
]

blacksmith_tree = [
    "p2_Padded Archer Armor", "p2_Fletching", "p2_Forging", "p2_Scale Barding Armor", "p2_Scale Mail Armor",
    "p2_Leather Archer Armor", "p2_Bodkin Arrow", "p2_Iron Casting", "p2_Chain Barding Armor", "p2_Chain Mail Armor",
    "p2_Ring Archer Armor", "p2_Bracer", "p2_Blast Furnace", "p2_Plate Barding Armor", "p2_Plate Mail Armor"
]

dock_tree = [
    "p2_Gillnets","p2_War Galley",
    "p2_Fast Fire Ship", "p2_Heavy Demolition Ship", "p2_Galleon", "p2_Elite Turtle Ship", "p2_Elite Longboat", "p2_Elite Caravel",
    "p2_Elite Cannon Galleon","p2_Dry Dock","p2_Careening", "p2_Shipwright"
]
university_tree = [
    "p2_Masonry", "p2_Fortified Wall", "p2_Ballistics", "p2_Guard Tower", "p2_Heated Shot", "p2_Murder Holes", "p2_Treadmill Crane",
    "p2_Architecture", "p2_Chemistry", "p2_Siege Engineers", "p2_Keep", "p2_Arrowslits",
]
castle_tree = [
    "p2_Hoardings", "p2_Sappers", "p2_Conscription", "p2_Spies/Treason"
]

donjon_tree = [
    "p2_Elite Serjeant", "p2_Elite Konnik"
]

monastery_tree = [
    "p2_Redemption", "p2_Atonement", "p2_Herbal Medicine", "p2_Heresy", "p2_Sanctity", "p2_Fervor",
    "p2_Illumination", "p2_Block Printing", "p2_Faith", "p2_Theocracy"
]

town_center_tree = [
    "p2_Feudal Age", "p2_Loom",
    "p2_Town Watch", "p2_Castle Age", "p2_Wheelbarrow",
    "p2_Town Patrol", "p2_Imperial Age", "p2_Hand Cart"
]

market_tree = [
    "p2_Gold Mining", "p2_Stone Mining", "p2_Double-Bit Axe",
    "p2_Gold Shaft Mining", "p2_Stone Shaft Mining", "p2_Bow Saw", "p2_Coinage", "p2_Caravan", "p2_Heavy Plow",
    "p2_Two-Man Saw", "p2_Banking", "p2_Guilds", "p2_Crop Rotation",
    "p2_Horse Collar"
]


# Archery Range Tree
df['p2_archery_range_tree'] = df[archery_range_tree].sum(axis=1)
df.drop(archery_range_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_archery_range_tree')
df.insert(31, "p2_archery_range_tree", col_matchup)

# Barracks Tree
df['p2_barracks_tree'] = df[barracks_tree].sum(axis=1)
df.drop(barracks_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_barracks_tree')
df.insert(31, "p2_barracks_tree", col_matchup)

# Stable Tree
df['p2_stable_tree'] = df[stable_tree].sum(axis=1)
df.drop(stable_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_stable_tree')
df.insert(31, "p2_stable_tree", col_matchup)

# Siege Workshop Tree
df['p2_siege_workshop_tree'] = df[siege_workshop_tree].sum(axis=1)
df.drop(siege_workshop_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_siege_workshop_tree')
df.insert(31, "p2_siege_workshop_tree", col_matchup)

# Blacksmith Tree
df['p2_blacksmith_tree'] = df[blacksmith_tree].sum(axis=1)
df.drop(blacksmith_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_blacksmith_tree')
df.insert(31, "p2_blacksmith_tree", col_matchup)

# Dock Tree
df['p2_dock_tree'] = df[dock_tree].sum(axis=1)
df.drop(dock_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_dock_tree')
df.insert(31, "p2_dock_tree", col_matchup)

# University Tree
df['p2_university_tree'] = df[university_tree].sum(axis=1)
df.drop(university_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_university_tree')
df.insert(31, "p2_university_tree", col_matchup)

# Castle Tree
df['p2_castle_tree'] = df[castle_tree].sum(axis=1)
df.drop(castle_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_castle_tree')
df.insert(31, "p2_castle_tree", col_matchup)

# Donjon Tree
df['p2_donjon_tree'] = df[donjon_tree].sum(axis=1)
df.drop(donjon_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_donjon_tree')
df.insert(31, "p2_donjon_tree", col_matchup)

# Monastery Tree
df['p2_monastery_tree'] = df[monastery_tree].sum(axis=1)
df.drop(monastery_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_monastery_tree')
df.insert(31, "p2_monastery_tree", col_matchup)

# Town Center Tree
df['p2_town_center_tree'] = df[town_center_tree].sum(axis=1)
df.drop(town_center_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_town_center_tree')
df.insert(31, "p2_town_center_tree", col_matchup)

# Market Tree
df['p2_market_tree'] = df[market_tree].sum(axis=1)
df.drop(market_tree, axis=1, inplace=True)
col_matchup = df.pop('p2_market_tree')
df.insert(31, "p2_market_tree", col_matchup)

#######################################################################
print("Removing low info gain columns...")

lowinfo_columns = ["p2_archery_range_tree", "p2_building_defensive", "p2_unit_gunpowder", "p1_unit_gunpowder", 
"p1_archery_range_tree", "p1_building_defensive", "p1_castle_tree", "p2_castle_tree", "p1_monastery_tree",
"p1_siege_workshop_tree", "p2_siege_workshop_tree", "p2_monastery_tree", "game_alive", "current_gametime",
"Unnamed: 0", "civ_matchup","p2_unit_ship","p1_unit_ship","p2_donjon_tree","p1_donjon_tree","p2_dock_tree","p1_dock_tree"]

df.drop(lowinfo_columns, axis=1, inplace=True)

print("Removing irrevelant columns...")
irrevelant_columns = [
    "p1_Stirrups", "p1_Royal Heirs", "p1_Anarchy", "p1_Elite Huskarl", "p1_Corvinian Army", 
    "p1_Tigui", "p1_Drill", "p1_Elite Gbeto", "p1_Howdah", "p1_El Dorado", "p1_Marauders", 
    "p1_Chieftains", "p1_Bearded Axe", "p1_Orthodoxy", "p1_Druzhina", "p1_Tower Shields", 
    "p1_Eupseong", "p1_Elite Mangudai", "p1_Elite Teutonic Knight", "p1_Pavise", 
    "p1_Elite Janissary", "p1_Yasama", "p1_Kataparuto", "p1_Elite Rattan Archer", "p1_Chatras", 
    "p1_Paper Money", "p1_Atlatl", "p1_Yeomen", "p1_Elite Longbowman", "p1_Elite Jaguar Warrior", 
    "p1_Garland Wars", "p1_", "p1_Elite Obuch", "p1_Crenellations", "p1_Hul'che Javelineers", 
    "p1_Fabric Shields", "p1_Steppe Husbandry", "p1_Perfusion", "p1_Elite Berserk", "p1_Elite Boyar", 
    "p1_Heavy Cav Archer", "p1_Chivalry", "p1_Elite Plumed Archer", "p1_Silk Armor", 
    "p1_Manipur Cavalry", "p1_First Crusade", "p1_Warwolf", "p1_Kamandaran", "p1_Maghrebi Camels", 
    "p1_Elite Conquistador", "p1_Farimba", "p1_Elite Kamayuk", "p1_Elite Woad Raider", 
    "p1_Elite Throwing Axeman", "p1_Recurve Bow", "p1_Arquebus", "p1_Hill Forts", 
    "p1_Elite Hussite Wagon", "p1_Elite Leitis", "p1_Tusk Swords", "p1_Double Crossbow", 
    "p1_Elite Ballista Elephant", "p1_Elite Tarkan", "p1_Elite Samurai", "p1_Shinkichon", 
    "p1_Elite Keshik", "p1_Supremacy", "p1_Torsion Engines", "p1_Elite Magyar Huszar", 
    "p1_Elite Arambai", "p1_Zealotry", "p1_Elite Mameluke", "p1_Elite Camel Archer", "p1_Kasbah", 
    "p1_Bagains", "p1_Elite Steppe Lancer", "p1_Ironclad", "p1_Forced Levy", "p1_Stronghold", 
    "p1_Elite Cataphract", "p1_Logistica", "p1_Elite Organ Gun", "p1_Elite Shotel Warrior", 
    "p1_Elite Genoese Crossbowman", "p1_Elite Chu Ko Nu", "p1_Artillery", "p1_Elite War Wagon", 
    "p1_Flemish Revolution", "p1_Burgundian Vineyards", "p1_Timurid Siegecraft", "p1_Sipahi", 
    "p1_Berserkergang", "p1_Andean Sling", "p1_Mahouts", "p1_Nomads", "p1_Elite Coustillier", 
    "p1_Sultans", "p1_Thalassocracy", "p1_Rocketry", "p1_Furor Celtica", "p1_Scutage", 
    "p1_Inquisition", "p1_Atheism", "p1_Shatagni", "p1_Elite Karambit Warrior", 
    "p1_Elite War Elephant", "p1_Carrack", "p1_Greek Fire", "p1_Great Wall", "p1_Cuman Mercenaries", 
    "p1_Madrasah", "p1_Silk Road",
    
    "p2_Stirrups", "p2_Royal Heirs", "p2_Anarchy", "p2_Elite Huskarl", "p2_Corvinian Army", 
    "p2_Tigui", "p2_Drill", "p2_Elite Gbeto", "p2_Howdah", "p2_El Dorado", "p2_Marauders", 
    "p2_Chieftains", "p2_Bearded Axe", "p2_Orthodoxy", "p2_Druzhina", "p2_Tower Shields", 
    "p2_Eupseong", "p2_Elite Mangudai", "p2_Elite Teutonic Knight", "p2_Pavise", 
    "p2_Elite Janissary", "p2_Yasama", "p2_Kataparuto", "p2_Elite Rattan Archer", "p2_Chatras", 
    "p2_Paper Money", "p2_Atlatl", "p2_Yeomen", "p2_Elite Longbowman", "p2_Elite Jaguar Warrior", 
    "p2_Garland Wars", "p2_", "p2_Elite Obuch", "p2_Crenellations", "p2_Hul'che Javelineers", 
    "p2_Fabric Shields", "p2_Steppe Husbandry", "p2_Perfusion", "p2_Elite Berserk", "p2_Elite Boyar", 
    "p2_Heavy Cav Archer", "p2_Chivalry", "p2_Elite Plumed Archer", "p2_Silk Armor", 
    "p2_Manipur Cavalry", "p2_First Crusade", "p2_Warwolf", "p2_Kamandaran", "p2_Maghrebi Camels", 
    "p2_Elite Conquistador", "p2_Farimba", "p2_Elite Kamayuk", "p2_Elite Woad Raider", 
    "p2_Elite Throwing Axeman", "p2_Recurve Bow", "p2_Arquebus", "p2_Hill Forts", 
    "p2_Elite Hussite Wagon", "p2_Elite Leitis", "p2_Tusk Swords", "p2_Double Crossbow", 
    "p2_Elite Ballista Elephant", "p2_Elite Tarkan", "p2_Elite Samurai", "p2_Shinkichon", 
    "p2_Elite Keshik", "p2_Supremacy", "p2_Torsion Engines", "p2_Elite Magyar Huszar", 
    "p2_Elite Arambai", "p2_Zealotry", "p2_Elite Mameluke", "p2_Elite Camel Archer", "p2_Kasbah", 
    "p2_Bagains", "p2_Elite Steppe Lancer", "p2_Ironclad", "p2_Forced Levy", "p2_Stronghold", 
    "p2_Elite Cataphract", "p2_Logistica", "p2_Elite Organ Gun", "p2_Elite Shotel Warrior", 
    "p2_Elite Genoese Crossbowman", "p2_Elite Chu Ko Nu", "p2_Artillery", "p2_Elite War Wagon", 
    "p2_Flemish Revolution", "p2_Burgundian Vineyards", "p2_Timurid Siegecraft", "p2_Sipahi", 
    "p2_Berserkergang", "p2_Andean Sling", "p2_Mahouts", "p2_Nomads", "p2_Elite Coustillier", 
    "p2_Sultans", "p2_Thalassocracy", "p2_Rocketry", "p2_Furor Celtica", "p2_Scutage", 
    "p2_Inquisition", "p2_Atheism", "p2_Shatagni", "p2_Elite Karambit Warrior", 
    "p2_Elite War Elephant", "p2_Carrack", "p2_Greek Fire", "p2_Great Wall", "p2_Cuman Mercenaries", 
    "p2_Madrasah", "p2_Silk Road"
]

df.drop(irrevelant_columns, axis=1, inplace=True)

# Define percentiles
lower_percentile = 1
upper_percentile = 99

# Replace extreme values with the specified percentiles
for column in df.columns:
    lower = df[column].quantile(lower_percentile / 100)
    upper = df[column].quantile(upper_percentile / 100)
    
    # Apply changes
    df[column] = df[column].clip(lower, upper)

df = df.loc[df['p1 Castle Age Time'] != 0]
df = df.loc[df['p2 Castle Age Time'] != 0]
df = df.loc[df['p1_unit_infantry'] != 0]
df = df.loc[df['p2_unit_infantry'] != 0]

columns_to_drop = []

# Identify matching columns with p1_ and p2_ prefixes
for col in df.columns:
    if col.startswith('p1_'):
        # Derive the matching p2_ column and diff_ column name
        suffix = col[3:]  # Get the part after 'p1_'
        p2_col = f'p2_{suffix}'
        diff_col = f'diff_{suffix}'
        
        # Check if the matching p2_ column exists
        if p2_col in df.columns:
            # Calculate the difference and create the diff_ column
            df[diff_col] = df[col] - df[p2_col]
            
            # Mark p1_ and p2_ columns for dropping
            columns_to_drop.extend([col, p2_col])

# Identify matching columns with p1_ and p2_ prefixes
for col in df.columns:
    if col.startswith('p1 '):  # Looking for columns that start with 'p1 '
        # Derive the matching p2_ column and diff_ column name
        suffix = col[3:]  # Get the part after 'p1 '
        p2_col = f'p2 {suffix}'
        diff_col = f'diff {suffix}'
        
        # Check if the matching p2_ column exists
        if p2_col in df.columns:
            # Calculate the difference and create the diff_ column
            df[diff_col] = df[col] - df[p2_col]
            
            # Mark p1_ and p2_ columns for dropping
            columns_to_drop.extend([col, p2_col])


# Drop the original p1_ and p2_ columns
df = df.drop(columns=columns_to_drop)
df = df.drop(columns="avg_elo")


columns_to_normalize = df.columns[df.columns != 'winner']

# Apply min-max normalization manually
df[columns_to_normalize] = df[columns_to_normalize].apply(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

print("Rounding values...")
df=df.round(5)

df.to_csv(input_csv, index=False)

print("Creating data visuals...")
correlation_matrix = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', vmin=-1, vmax=1, linewidths=0.3, annot_kws={'size': 8})
# Rotate x and y labels for better visibility

# Adjust layout to ensure everything fits
plt.tight_layout()
plt.title('Correlation Heatmap')
plt.savefig('Correlation.png')

df.hist(bins=100, figsize=(18, 16), grid=False, edgecolor='black',color='purple', alpha=0.7)

# Show the plots
plt.tight_layout()
plt.savefig('Histogram.png')

# Function to calculate entropy
def entropy(column):
    probabilities = column.value_counts(normalize=True)
    return -sum(probabilities * np.log2(probabilities))

# Function to calculate Information Gain
def information_gain(df, target_column, feature_column):
    # Calculate entropy of the whole dataset
    total_entropy = entropy(df[target_column])
    
    # Calculate the weighted entropy for each value in the feature column
    values = df[feature_column].unique()
    weighted_entropy = 0
    
    for value in values:
        subset = df[df[feature_column] == value]
        weighted_entropy += (len(subset) / len(df)) * entropy(subset[target_column])
    
    # Information Gain
    return total_entropy - weighted_entropy

# Calculate IG for all features
target_column = 'winner'
features = [col for col in df.columns if col != target_column]

# Calculate and print IG for each feature
ig_values = {}
for feature in features:
    ig_values[feature] = information_gain(df, target_column, feature)

# Display Information Gain for each feature
for feature, ig in ig_values.items():
    print(f"Information Gain for '{feature}': {ig}")

# Visualize the information gain of each feature
ig_df = pd.DataFrame({
    'Feature': list(ig_values.keys()),
    'Information Gain': list(ig_values.values())
})

# Sort by Information Gain for better visualization
ig_df = ig_df.sort_values(by='Information Gain', ascending=False)

# Adjusting the gradient color palette to range from red to yellow for each row
colors = sns.color_palette("YlOrRd", len(ig_df))

# Create a horizontal bar plot with the new gradient colors
plt.figure(figsize=(10, 6))
sns.barplot(
    x='Information Gain', 
    y='Feature', 
    data=ig_df, 
    palette="rainbow"
)
plt.title('Information Gain by Feature', fontsize=14)
plt.xlabel('Information Gain', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()

plt.savefig('infogain.png')

print("Preprocessing is done, training models **************************************************************************")

data = pd.read_csv(input_csv)

# Randomly sample 20% of the data for efficient training
data = data.sample(frac=0.2, random_state=42)

# Define features (X) and target (y)
X = data.drop(columns=['winner'])
y = data['winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the Bayesian classifier
model = GaussianNB()

# Define hyperparameter grid for tuning
param_grid = {'var_smoothing': np.logspace(-9, -1, 10)}

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,  # 10-fold cross-validation
    verbose=0  # Show detailed training progress
)

# Fit the model and find the best parameters
grid_search.fit(X_train, y_train)

# Display the best hyperparameters and the corresponding score
print("\n","Results for Naive Bayes: ","\n")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate the final model with the best parameters
best_model = grid_search.best_estimator_

# Perform predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate and display train-test split metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

# ROC Curve and AUC for Train-Test Split
y_score = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)


print("\nTrain-Test Split Metrics:")
metrics_table_test = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
print(metrics_table_test,"\n")

# Perform cross-validation and get predictions
y_cv_pred = cross_val_predict(best_model, X, y, cv=10, method="predict")
y_cv_proba = cross_val_predict(best_model, X, y, cv=10, method="predict_proba")[:, 1]  # For AUC calculation

# Calculate metrics for cross-validation
cv_accuracy = accuracy_score(y, y_cv_pred)
cv_precision = precision_score(y, y_cv_pred, average='binary')
cv_recall = recall_score(y, y_cv_pred, average='binary')
cv_f1 = f1_score(y, y_cv_pred, average='binary')
fpr, tpr, _ = roc_curve(y, y_cv_proba)
cv_auc = auc(fpr, tpr)

# Display metrics for cross-validation
cv_metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Cross-Validation Score': [cv_accuracy, cv_precision, cv_recall, cv_f1, cv_auc]
})
print("Cross-Validation Metrics:")
print(cv_metrics_table)



plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Train-Test Split)')
plt.legend(loc='lower right')
plt.savefig('Naive_Bayes_ROC.png')


# Load the CSV into a pandas DataFrame
data = pd.read_csv(input_csv)

# Randomly sample 10% of the data for efficiency
data = data.sample(frac=0.1, random_state=42)

# Split into features and target
X = data.drop(columns=['winner'])
y = data['winner']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Define hyperparameter grid for tuning
param_grid = {
    'n_neighbors': [3, 5, 10, 50, 100],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'cosine'],
    'algorithm': ['auto']
}

# Reduced grid search for time-efficiency
reduced_param_grid = {
    'n_neighbors': [50, 100],
    'weights': ['uniform'],
    'metric': ['euclidean', 'manhattan'],
    'algorithm': ['auto']
}

# Set up GridSearchCV with 10-fold cross-validation
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=reduced_param_grid,
    cv=10,
    n_jobs=-1,
    verbose=0,
    scoring='accuracy'
)

# Fit the grid search model to the training data
grid_search.fit(X_train, y_train)

print("\n","Results for K-Nearest Neighbors: ","\n")
# Display the best hyperparameters and corresponding score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Get the best model from grid search
best_knn = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_knn.predict(X_test)
y_prob = best_knn.predict_proba(X_test)[:, 1]

# Calculate metrics for train-test split
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Display metrics for train-test split
print("\nTrain-Test Split Metrics:")
metrics_table_split = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Score': [accuracy, precision, recall, f1, roc_auc]
})
print(metrics_table_split)


plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('K_Nearest_ROC.png')


# Perform 10-fold cross-validation and calculate metrics
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracy = cross_val_score(best_knn, X, y, cv=kf, scoring='accuracy')
cv_precision = cross_val_score(best_knn, X, y, cv=kf, scoring=make_scorer(precision_score))
cv_recall = cross_val_score(best_knn, X, y, cv=kf, scoring=make_scorer(recall_score))
cv_f1 = cross_val_score(best_knn, X, y, cv=kf, scoring=make_scorer(f1_score))
cv_auc = cross_val_score(best_knn, X, y, cv=kf, scoring='roc_auc')

# Display metrics for cross-validation
print("\nCross-Validation Metrics:")
metrics_table_cv = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC'],
    'Mean Score': [
        cv_accuracy.mean(),
        cv_precision.mean(),
        cv_recall.mean(),
        cv_f1.mean(),
        cv_auc.mean()
    ],
    'Std Dev': [
        cv_accuracy.std(),
        cv_precision.std(),
        cv_recall.std(),
        cv_f1.std(),
        cv_auc.std()
    ]
})
print(metrics_table_cv)


# Load dataset
data = pd.read_csv(input_csv)

# Randomly sample 1% of the data because grid search and cross-validation might take too long
data = data.sample(frac=0.01, random_state=42)

# Define features (X) and target (y)
X = data.drop(columns=['winner'])
y = data['winner']

# Standardize the features for the neural network model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Define and train the neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))  # Single output unit for binary classification
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',  # Binary cross-entropy loss
                  metrics=['accuracy'])
    return model

# Create and train the neural network
nn_model = create_nn_model()
history = nn_model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0
)

# 10-Fold Cross-Validation Metrics
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_accuracies = []
cv_precisions = []
cv_recalls = []
cv_f1_scores = []
cv_aucs = []

for train_idx, val_idx in kf.split(X_scaled):
    X_train_cv, X_val_cv = X_scaled[train_idx], X_scaled[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]  # Use .iloc to properly index 'y'

    # Create and train the model for each fold
    nn_model_cv = create_nn_model()
    nn_model_cv.fit(X_train_cv, y_train_cv, epochs=50, batch_size=32, verbose=0)

    # Predict and calculate metrics for the fold
    y_pred_proba_cv = nn_model_cv.predict(X_val_cv)
    y_pred_cv = (y_pred_proba_cv > 0.5).astype(int)
    
    accuracy_cv = accuracy_score(y_val_cv, y_pred_cv)
    precision_cv = precision_score(y_val_cv, y_pred_cv)
    recall_cv = recall_score(y_val_cv, y_pred_cv)
    f1_cv = f1_score(y_val_cv, y_pred_cv)
    fpr_cv, tpr_cv, _ = roc_curve(y_val_cv, y_pred_proba_cv)
    auc_cv = auc(fpr_cv, tpr_cv)

    # Append metrics for each fold
    cv_accuracies.append(accuracy_cv)
    cv_precisions.append(precision_cv)
    cv_recalls.append(recall_cv)
    cv_f1_scores.append(f1_cv)
    cv_aucs.append(auc_cv)

# Calculate the mean and standard deviation for the CV metrics
cv_metrics = {
    'Accuracy': [np.mean(cv_accuracies), np.std(cv_accuracies)],
    'Precision': [np.mean(cv_precisions), np.std(cv_precisions)],
    'Recall': [np.mean(cv_recalls), np.std(cv_recalls)],
    'F1 Score': [np.mean(cv_f1_scores), np.std(cv_f1_scores)],
    'AUC': [np.mean(cv_aucs), np.std(cv_aucs)],
}

print("\n","Results for Neural Network: ","\n")
cv_metrics_df = pd.DataFrame(cv_metrics, index=['Mean', 'Std Dev'])
print("\n10-Fold Cross-Validation Metrics:")
print(cv_metrics_df)

# Evaluate the neural network model on the test set (80-20 split)
y_pred_proba_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)  # Threshold at 0.5 for binary classification

# Calculate metrics for the neural network model on the test set
accuracy_nn = accuracy_score(y_test, y_pred_nn)
precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Display metrics in a table for neural network on test set
nn_metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Score': [accuracy_nn, precision_nn, recall_nn, f1_nn, roc_auc_nn]
})
print("\nTrain-Test Split Metrics:")
print(nn_metrics_table)


plt.figure()
plt.plot(fpr_nn, tpr_nn, color='blue', label=f"NN AUC = {roc_auc_nn:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Neural Network - Test Set)')
plt.legend(loc='lower right')
plt.savefig('Neural_ROC.png')

# Load dataset (replace 'output2.csv' with your dataset file)
data = pd.read_csv(input_csv)

# Randomly sample 1% of the data to reduce computation time
data = data.sample(frac=0.1, random_state=42)

# Split into features (X) and target (y)
X = data.drop(columns=['winner'])
y = data['winner']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

# Reduced grid search for time-efficiency
reduced_param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [10, 20],  # Maximum depth of trees
    'min_samples_split': [5, 10],  # Minimum samples required to split a node
    'min_samples_leaf': [2, 4],  # Minimum samples required to be at a leaf node
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}
# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=reduced_param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
    scoring='accuracy',
)

# Fit the grid search model to the training data
grid_search.fit(X_train, y_train)

# Display the best hyperparameters and the corresponding score
print("\n","Results for Random Forest: ","\n")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Get the best model from grid search
best_rf = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]  # For AUC calculation

# Calculate metrics on the test set
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average='binary')
test_recall = recall_score(y_test, y_pred, average='binary')
test_f1 = f1_score(y_test, y_pred, average='binary')
fpr, tpr, _ = roc_curve(y_test, y_proba)
test_auc = auc(fpr, tpr)

# Display metrics for the test set
test_metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Test Set Score': [test_accuracy, test_precision, test_recall, test_f1, test_auc],
})
print("\nTrain-Test Split Metrics:")
print(test_metrics_table)

# Plot ROC curve for the test set

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"AUC = {test_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.savefig('Random_Forest_ROC.png')

# Perform cross-validation predictions
y_cv_pred = cross_val_predict(best_rf, X, y, cv=5, method="predict")
y_cv_proba = cross_val_predict(best_rf, X, y, cv=5, method="predict_proba")[:, 1]  # For AUC

# Calculate metrics for cross-validation
cv_accuracy = accuracy_score(y, y_cv_pred)
cv_precision = precision_score(y, y_cv_pred, average='binary')
cv_recall = recall_score(y, y_cv_pred, average='binary')
cv_f1 = f1_score(y, y_cv_pred, average='binary')
fpr, tpr, _ = roc_curve(y, y_cv_proba)
cv_auc = auc(fpr, tpr)

# Display metrics for cross-validation
cv_metrics_table = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'],
    'Cross-Validation Score': [cv_accuracy, cv_precision, cv_recall, cv_f1, cv_auc],
})
print("\nCross-Validation Metrics:")
print(cv_metrics_table)


# Load your dataset
df = pd.read_csv(input_csv)

# Randomly sample 1% of the data to reduce computation time
df_sampled = df.sample(frac=0.01, random_state=42)

# Assuming 'winner' is the target variable and the rest are features
X = df_sampled.drop(columns=['winner'])  # Features
y = df_sampled['winner']  # Target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the SVM classifier
svm = SVC(probability=True, random_state=42)

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient
    'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],  # Different kernel types
    'degree': [2, 3, 4]  # Polynomial degree for 'poly' kernel
}

# Reduced grid search for time-efficiency
reduced_param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'gamma': ['auto', 1],  # Kernel coefficient
    'kernel': ['linear', 'poly'],  # Different kernel types
    'degree': [2, 3]  # Polynomial degree for 'poly' kernel
}

# Perform Grid Search with 10-fold cross-validation
grid_search = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1, verbose=0, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\n","Results for SVM:","\n")
# Print the best hyperparameters found by GridSearchCV
print(f"\nBest Hyperparameters: {grid_search.best_params_}")

# Train the model with the best hyperparameters
best_svm = grid_search.best_estimator_

# Perform 10-fold cross-validation on the entire dataset using the best model
cv_pred = cross_val_predict(best_svm, X, y, cv=10, method='predict')
cv_proba = cross_val_predict(best_svm, X, y, cv=10, method='predict_proba')[:, 1]

cv_accuracy = accuracy_score(y, cv_pred)
cv_precision = precision_score(y, cv_pred)
cv_recall = recall_score(y, cv_pred)
cv_f1 = f1_score(y, cv_pred)
fpr, tpr, _ = roc_curve(y, cv_proba)
cv_auc = roc_auc_score(y, cv_proba)

print("\n10-Fold Cross-Validation Metrics:")
print(f"Accuracy: {cv_accuracy:.2f}")
print(f"Precision: {cv_precision:.2f}")
print(f"Recall: {cv_recall:.2f}")
print(f"F1-Score: {cv_f1:.2f}")
print(f"AUC: {cv_auc:.2f}")

# Predict on the test set using the best model
y_pred = best_svm.predict(X_test)
y_proba = best_svm.predict_proba(X_test)[:, 1]

# Calculate the metrics for test data
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_proba)

# Display test set metrics
test_metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Test Set Score': [test_accuracy, test_precision, test_recall, test_f1, test_auc],
})
print("\nTrain-Test Split Metrics:")
print(test_metrics_df)


plt.figure()
plt.plot(fpr, tpr, color='blue', label=f"AUC = {test_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Test Set)')
plt.legend(loc='lower right')
plt.savefig('SVM_ROC.png')
