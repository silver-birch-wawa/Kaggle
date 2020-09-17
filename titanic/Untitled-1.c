#include<stdio.h>
struct date{
        int day;
        char month[20];
};
    
int main(){
    struct date dates1;
    dates1.day =12;
    dates1.month="Jan.";
    printf("%s",dates1.month);
}
