#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vectorAdd(int input1[], int input2[], int output[]){
  for(int i = 0; i < 1000; i++){
    output[i] = input1[i] + input2[i];
  }
}

int main(void){
  int arr1[1000];
  int arr2[1000];
  int sum[1000];
  int out = 0;
  srand(time(NULL));
  for(int i = 0; i < 1000; i++){
    arr1[i] = rand() % 1000;
    arr2[i] = rand() % 1000;
  }
  vectorAdd(arr1, arr2, sum);
  for(int i = 0; i < 1000; i++){
    out += sum[i];
  }
  printf("Output: %d \n", out % 1000);
  return 0;
}
