CC = gcc
CFLAGS = -pthread -Wall -g -mavx -mfma
TARGET = test


all: $(TARGET)

$(TARGET): testsgemm.o kernel_v1.o
	$(CC) $(CFLAGS) -o $(TARGET) testsgemm.o kernel_v1.o


testsgemm.o: testsgemm.c kernel_v1.h
	$(CC) $(CFLAGS) -c testsgemm.c


kernel_v1.o: kernel_v1.c kernel_v1.h
	$(CC) $(CFLAGS) -c kernel_v1.c


clean:
	rm -f $(TARGET) *.o