#include <bits/stdc++.h>

using namespace std;

struct abc{
    int a;
    int b;
    int c;
};


int main(){
    struct abc a = {1, 2, 3};
    struct abc *ptr = &a;

    

    // ptr->a equivalent to (*ptr).a = (*&a).a = a.a = 1
    cout << ptr->a << " " << ptr->b << " " << ptr->c << endl;

    return 0;
}
