# Client
This is the client class, this will be deployed on a Raspberry Pi.


# Pinout schema
| Signal Name       | Name on the microphone array | BCM GPIO Number | Physical Pin (BOARD) | Notes                        |
|-------------------|------------------------------|-----------------|----------------------|------------------------------|
| LED Clock         | **LED_SCLK**                 | **GPIO 11**     | **23**               | (SCLK on Pi)                 |
| LED Data          | **LED_MOSI**                 | **GPIO 10**     | **19**               | (MOSI on Pi)                 |
| Microphone Data 0 | **MIC_D0**                   | **GPIO 21**     | **40**               |                              |
| Microphone Data 1 | **MIC_D1**                   | **GPIO 20**     | **38**               |                              |
| Microphone Data 2 | **MIC_D2**                   | **GPIO 26**     | **37**               |                              |
| Microphone Data 3 | **MIC_D3**                   | **GPIO 16**     | **36**               |                              |
| Microphone WS     | **MIC_WS**                   | **GPIO 19**     | **35**               | (Word Select/LRCK)           |
| Microphone Clock  | **MIC_CK**                   | **GPIO 18**      | **12**               | PCM Clock  |
| Power Input       | **VIN** (3.3V)               | –               | **1**                | 3.3VDC supply                |
| Ground            | **GND**                      | –               | **6**                | Common ground                |
