CC = gcc

CFLAGS=-Wall -g
ZFLAGS=-Wall -g -O0
TFLAGS=-Wall -g -O3

RM = rm -rf

TARGET = 

$(TARGET): $(TARGET).c
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c

ef0:
	$(CC) $(ZFLAGS) -o $(TARGET) $(TARGET).c

ef3:
	$(CC) $(TFLAGS) -o $(TARGET) $(TARGET).c
