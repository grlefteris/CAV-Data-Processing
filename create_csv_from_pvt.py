import geofun as g
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('ggplot')

save_csv = True

path = r'D:\Apps\CF_Data_Processing\Cherasco 30Aug18\test 1\COM3_180830_065155.Pvt' #r'C:\Users\mattako\Desktop\parse and geo/'

file = path# + 'COM3_180504_101253.Pvt'

head = 'ToW,Latitude,Longitude,Height,MSL,Horizontal accuracy,Vertical accuracy,Vn,Ve,Vd,ground speed,heading,speed accuracy,heading accuracy'.split(',')
df = pd.read_csv(file, delimiter= ' ',header=None, names = head)

# % 1 - ToW in ms
# % 2 - Latitude in deg
# % 3 - Longitude in deg
# % 4 - Height
# % 5 - MSL mean sea level
# % 6 - Horizontal accuracy
# % 7 - Vertical accuracy
# % 8 - Vn in m
# % 9 - Ve in m
# % 10 - Vd in m
# % 11 - ground speed in m
# % 12 - heading in deg
# % 13 - speed accuracy
# % 14 - heading

df['Latitude'],df['Longitude'],df['Height'] = df['Latitude']/180*np.pi,df['Longitude']/180*np.pi,df['Height']/180*np.pi

center = [0.7930176586258354,0.14891952130512026,4.229708182402579]
# center=[df['Latitude'][0],df['Longitude'][0],df['Height'][0]]

# df['E'],df['N'],df['U'] = g.geo2enu(df['Latitude'],df['Longitude'],df['Height'], df['Latitude'].mean(),df['Longitude'].mean(),df['Height'].mean() )
df['E'],df['N'],df['U'] = g.geo2enu(df['Latitude'],df['Longitude'],df['Height'], center[0],center[1],center[2] )

# print(df['Latitude'].mean(),df['Longitude'].mean(),df['Height'].mean())

df.loc[:,'Timestamp'] = df['ToW'] - df.loc[0,'ToW']
df.loc[:,'Timestamp'] = df['Timestamp']/1000

### GNSS calculates from the start of week. We need from the start of the day (in Italy so +2)
df['sec_from_midnight'] =  df['ToW']/1000%(3600*24)+3600*2

# c_spn = 'Speed calculated from position'
# df[c_spn] = 0
c_dx = 'Speed from position'
df[c_dx] = 0

df = df.reset_index()

plt.figure()
plt.title('Speed profile')
time = df['Timestamp']
df['ground speed'] = df['ground speed']/10
speed = df['ground speed']
plt.plot(time,speed,'-xb')
plt.xlabel('Time in s')
plt.ylabel('Speed in km/h')

for index, row in df.iterrows():
    if index!=0:
        df.loc[index,c_dx] = np.sqrt( np.power((row['E']-df.loc[index-1,'E']),2) + np.power((row['N']-df.loc[index-1,'N']),2) + np.power((row['U']-df.loc[index-1,'U']),2) ) /(df.loc[index,'Timestamp'] -df.loc[index-1,'Timestamp'] )

# plt.figure()
# plt.title('Speed profile from distance')
# plt.plot(time,df[c_dx],'-o')
#
# plt.figure()
# plt.plot(time,speed - df[c_dx] )

if save_csv:
    df.to_csv(file[:-4]+'new.csv')
# plt.show()