kdd99_classification_dict = {
    'NORMAL': ('normal',),
    'PROBE': ('ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan',),
    'DOS': ('apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm',),
    'U2R': ('buffer_overflow', 'httptunnel', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm',),
    'R2L': (
        'ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess',
        'spy',
        'warezclient', 'warezmaster', 'worm', 'xlock', 'xsnoop',),
}


def get_kdd99_big_classification(label):
    for classification in kdd99_classification_dict.keys():
        if kdd99_classification_dict[classification].__contains__(label):
            return classification
