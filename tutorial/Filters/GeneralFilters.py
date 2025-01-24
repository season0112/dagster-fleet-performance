def FilterOnPumpRunningOnly(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([2])].index, inplace=True)

def FilterOnPumpRunning(df_load):
    df_load.drop(df_load[~df_load['qc_supervisoryControlMode'].isin([2, 3])].index, inplace=True)

def FilterOnAllNull(df_load):
    df_load.dropna(how='all', inplace=True)

def FilterOnTestRigs(df_load):
    TestRigs = [
        "CIC-742f9eba-8f69-5efe-8f2c-52afeb2fb9d7",
        "CIC-59d1d85f-ef2c-5733-ac76-9ff2c2affd21",
        "CIC-b456cd2e-d054-5314-96e0-c06e38aac609",
        "CIC-6bff667e-4266-516f-877c-5f8e5eb8fb03",
        "CIC-85cb4dde-ed74-5b6a-bd9c-4b8b957728b2",
        "CIC-d19f89d1-cf41-58c5-a34d-74c1410f42a1",
        "CIC-b2369d60-5ae3-59b7-b5c8-f450161ee0ca",
        "CIC-46f31f71-9b6d-5af5-ba63-1c71d639dcc5",
        "CIC-ccd15bc7-bcff-5888-a9db-561a973686f2", 
        "CIC-3ce07698-2fb2-55ff-8200-8575e9cbac12",
        "CIC-691989e5-24b7-59b9-9807-537c89035f25",
        "CIC-991ce3a9-772d-5d54-ba67-ff6e1699c225",
        "CIC-ada0cb64-5a9f-5ace-b7d1-33ea05b65990", 
        "CIC-89c072c1-91fa-5952-91e8-6bb5120f4a14",
        "CIC-f42058e3-44c5-5d70-809d-f2ee78b2abf9",
        "CIC-d0951c0d-3601-5d73-871c-cc9f477c9d4e"
    ]
    indices_to_drop = df_load[df_load['clientid'].isin(TestRigs)].index
    df_load.drop(indices_to_drop, inplace=True)
    
def FilterOnInactiveCiC(df_load):
    InactiveCiC = [
        "CIC-412a00af-b272-59e4-b54c-a92ae19ae737",
        "CIC-414dcc91-5d43-5c68-bb0c-99f6242db6ec",
        "CIC-ab4b98b2-edd5-5249-93b5-5c10731aa73a",
        "CIC-ef39d4d1-8280-584e-ba7c-61bdc7a6c3cb",
        "CIC-ef51fea4-e871-5c9e-ad75-0b65495a590d",
        "CIC-ef6d4cf5-ec22-5ac0-9531-755c963e79f6",
        "CIC-ef90772f-091a-56bf-8d5a-318b8c95eae2",
        "CIC-efc4701d-9dfe-5ff7-a20a-9e9a29607ee8",
        "CIC-efcfcbbf-989e-5bf3-b9bf-384f8eab6e77",
        "CIC-717d3296-9a78-5365-aea3-2f30708782d9",
        "CIC-718dce53-6d73-57de-934a-561e4fa9d52c",
        "CIC-71e1908f-37a1-5740-8f39-1e77fa12031b",
        "CIC-72d30f8a-44cc-5c12-85e4-8cc39ad5888e",
        "CIC-a22696bd-e3d4-5bdc-a00b-77e27942644d",
        "CIC-a24295d3-1573-53df-9cab-e827e08d1836",
        "CIC-a24fcdeb-7dde-5353-9e0e-fc237848b4f9",
        "CIC-a2508ddf-f352-53d8-9f36-a54d9a43fbf1",
        "CIC-b1492ad5-3641-5dca-85f7-97a37615c1ea",
        "CIC-b2a2544e-2042-5b00-944f-81d39d442e1b",
        "CIC-c1b59804-4243-5ce2-b3ce-02bf87de49e0",
        "CIC-c20084b0-3558-5118-8f41-e5bd06b1c05e",
        "CIC-c3320531-6a05-5c91-8bee-55badc99bfbf",
        "CIC-c4bc47f0-09e3-532a-991a-931427950d5b",
        "CIC-d3e81003-bbb0-5780-bc77-7699d2cad385"
    ]
    indices_to_drop = df_load[df_load['clientid'].isin(InactiveCiC)].index
    df_load.drop(indices_to_drop, inplace=True)

def FilterOnDropZero(df_load):
    DropZeroCiC = [
        "CIC-7b095e33-dc04-5b03-9dcb-f06bd0418ec0",
        "CIC-c6c44f70-ecc3-5d6b-927e-9abf8182eed1",
        "CIC-e7fff772-d411-541c-85eb-891dcd5e7ce1",
        "CIC-be2fad7a-0c29-5b3e-b86e-626f76636911",
        "CIC-6fef40b2-4c34-53b2-b23c-9712a930c5b3",
        "CIC-ebb95b3c-7edc-5eb6-88ec-465d8eff2b02",
        "CIC-8eec87e8-0c6d-5e84-832b-5c3455d02e13",
        "CIC-8fec91a4-44fa-5be0-b22c-2367412971c9",
        "CIC-6693d323-7bd3-54ed-a819-db8f41bbb07f",
        "CIC-d439551e-db7c-5423-8012-04c248a41fdc",
        "CIC-6fef40b2-4c34-53b2-b23c-9712a930c5b3",
        "CIC-c79e3db2-3982-5380-be58-df084d97aa11",
        "CIC-32aab2e2-52b3-5357-86d9-ead9d88b3772",
        "CIC-521458c2-fc25-5757-b5aa-f4f252c7825f",
        "CIC-ac259c01-b3f6-597d-b649-c2d4c228a980"
    ]
    indices_to_drop = df_load[df_load['clientid'].isin(DropZeroCiC)].index
    df_load.drop(indices_to_drop, inplace=True)

def FilterOnDropABit(df_load):
    DropABitCiC = [
        "CIC-25d6257f-5ab8-5e13-8a8a-bc70c7516cec",
        "CIC-0aa8677c-b545-5a12-86cb-8549c30f699b",
        "CIC-928d0ce0-69e4-5f6e-bad5-1603bc99c654",
        "CIC-4fccd29f-2778-5291-a166-4e3196c93731",
        "CIC-b578f3b1-13f3-59d8-aa99-01c9b036a3a1",
        "CIC-8d37426b-3c53-5da8-a256-01e63268e3d6",
        "CIC-50fc18d7-a02a-5a90-a60a-45ae9268fd01",
        "CIC-449016e3-ebc8-50de-9d60-1774ab665d0a",
        "CIC-7afbd4bf-3518-521d-9520-93a1a0449687",
        "CIC-24cf2672-96aa-5e59-a9a4-b4976d7e0311",
        "CIC-ea627084-0646-5f39-9d02-d5130a0e8183"
    ]
    indices_to_drop = df_load[df_load['clientid'].isin(DropABitCiC)].index
    df_load.drop(indices_to_drop, inplace=True)

def FilterOnIncreaseCounter(df_load): 
    IncreaseCiC = [
        "CIC-9368bfef-7eca-5bda-9a90-8d5a4be375c6",
        "CIC-7eede49c-42c2-5b41-94aa-481dad189abf",
        "CIC-cac39a52-37c2-55a6-a281-1d14c41a4325",
        "CIC-e265a6ef-8365-5bab-a661-c23935c3c6ea"
    ]
    indices_to_drop = df_load[df_load['clientid'].isin(IncreaseCiC)].index
    df_load.drop(indices_to_drop, inplace=True)


