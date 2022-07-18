import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import Normalizer

np.random.seed(777)

col_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
              'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
              'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
              'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
              'num_access_files', 'num_outbound_cmds', 'is_host_login',
              'is_guest_login', 'count', 'srv_count', 'serror_rate',
              'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
              'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
              'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
              'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
              'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
              'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level']


train_data = pd.read_csv('/KDDTrain%2B.csv', header = None, names = col_names, index_col = False)
test_data = pd.read_csv('https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.csv', header = None, names = col_names, index_col = False)

df = pd.concat([train_data, test_data]).drop('difficulty_level', 1)
df.shape

label_encoder = LabelEncoder()
for column in ['protocol_type','service','flag']:
  label_encoder.fit(df[column])
  df[column] = label_encoder.transform(df[column])


# lists to hold attack classifications
dos_attacks = ['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm']
probe_attacks = ['ipsweep','mscan','nmap','portsweep','saint','satan']
privilege_attacks = ['buffer_overflow','loadmdoule','perl','ps','rootkit','sqlattack','xterm']
access_attacks = ['ftp_write','guess_passwd','http_tunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xclock','xsnoop']

def map_attack(attack):
    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0
        
    return attack_type

# map the data and join to the data set
attack_map = df.attack_type.apply(map_attack)
df['attack_map'] = attack_map

train_df, test_df = train_test_split(df)

X = train_df.iloc[:,0:41]
Y = train_df.iloc[:,42]
T = test_df.iloc[:,0:41]
C = test_df.iloc[:,42]

trainX = np.array(X)
testT = np.array(T)

trainX.astype(float)
testT.astype(float)

y_train1 = np.array(Y)
y_test1= np.array(C)
y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)

scaler = Normalizer().fit(trainX)
trainX = scaler.transform(trainX)

scaler = Normalizer().fit(testT)
testT = scaler.transform(testT)

np.save('trainX.npy', trainX)
np.save('testT.npy', testT) 
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test) 