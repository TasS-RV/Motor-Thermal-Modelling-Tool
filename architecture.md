graph LR
    %% Styles
    classDef hv fill:#ADD8E6,stroke:#00008B,stroke-width:2px;
    classDef lv fill:#FFDAB9,stroke:#FF8C00,stroke-width:2px;
    classDef sig fill:#C1FFC1,stroke:#006400,stroke-width:2px;
    classDef grey fill:#F0F0F0,stroke:gray,stroke-dasharray: 5 5,color:gray;

    %% Nodes
    Input(270V Input) --> HV_Filter[HV INPUT FILTER<br>CMC + TVS + Y-Caps]:::hv
    HV_Filter --> DClink(DC Link):::grey
    DClink --> Inverter(IGBT Inverter):::grey
    Inverter --> Motor((Motor)):::grey
    
    %% LV Path
    DClink -.-> Buck(Isolated Buck<br>270V -> 24V):::grey
    Buck --> LV_Filter[LV INTERSTAGE<br>Ripple Filter]:::lv
    LV_Filter --> GateDrivers(Gate Drivers):::grey
    GateDrivers -.-> Inverter

    %% Signal Path
    Buck -.-> MCU(MCU / Logic):::grey
    
    Encoder((Encoder)):::grey --> Sig_Filter_In[SIGNAL FILTER<br>RC Network]:::sig
    Sig_Filter_In --> MCU
    
    MCU --> Sig_Filter_Out[OUTPUT FILTER<br>Ferrite Beads]:::sig
    Sig_Filter_Out --> Output(Analog Output<br>4-20mA)