/*********************************************************************/
/*                                                                   */
/* Tester for DSA555 assignment 1, part 1                            */
/* Version 1.0                                                       */
/* To compile:                                                       */
/*                                                                   */
/*      g++ a1q1main.cpp -std=c++0x                                  */
/*                                                                   */
/*********************************************************************/

#include "a1q1start.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
using namespace std;
//change the #define VERBOSE to 1 for a bit more error information
#define VERBOSE 0

struct Record{
	char word_[30];
	int count_;
};
ostream& operator<<(ostream& os, const Record rec){
	os << rec.word_;
	return os;
}
bool operator==(const Record& a,const Record& b){
	bool rc=false;
	if(strcmp(a.word_,b.word_)==0 && a.count_==b.count_){
		rc=true;
	}
	return rc;
}
bool operator!=(const Record& a,const Record& b){
	return !(a==b);
}
bool operator <(const Record& a, const Record& b){
	bool rc=false;
	if(strcmp(a.word_,b.word_) < 0){
		rc=true;
	}
	return rc;
}

template <typename T>
void removeItem(T arr[],int idx,int sz);
template <typename T>
bool checkList(const DList<T>& list,const T arr2[],int sz);
template <typename T>
void duplicateArray(T dest[], const T src[],int sz);
template <typename T>
void printLists(const DList<T>& list,const T array[],int sz);
template<typename T>
typename DList<T>::iterator setIterator(DList<T>& list, int idx);
template <typename T>
void addFront(const T& data,T arr[],int sz);
template <typename T>
void addBack(const T& data,T arr[],int sz);
template <typename T>
void add(const T& data,int idx, T arr[],int sz);

int main(void){
	const int cap=10000;
	const int numSearches=200;
	Record data[20]={
		{"the",1},
		{"quick",2},
		{"brown ",3},
		{"fox",4},
		{"jumped",5},
		{"over",6},
		{"lazy",7},
		{"dog",8},
		{"Calvin",9},
		{"and",10},
		{"Hobbes",11},
		{"night",12},
		{"watch",13},
		{"captain",14},
		{"carrot",15},
		{"lilac",16},
		{"lavender",17},
		{"lily",18},
		{"coffee",19},
		{"tea",20}
	};
	int intData[cap];

	//these array will mirror what happens to LL
	Record mirror[20];
	int intMirror[cap];
	for(int i=0;i<cap;i++){
		intData[cap-1-i]=rand();
		intMirror[cap-1-i]=intData[cap-1-i];
	}

	DList<Record> recList;
	DList<int> intList;

	bool passtest=true;
	int numPassed=0;



	/* Test constructors, begin and end functions*/
	cout << "test 1: create list empty list, check begin() and end()" << endl;
	if((recList.begin() != recList.end()) || 
		(intList.begin() != intList.end())){
		cout << "error 1: check your constructor, begin() and end() functions" << endl;
		passtest=false;
	}
	else{
		numPassed++;
	}

	if(passtest){
		for(int i=0;i<5;i++){
			addFront(data[i],mirror,i);
			recList.push_front(data[i]);
		}
		for(int i=5;i<10;i++){
			addBack(data[i],mirror,i);
			recList.push_back(data[i]);
		}
		DList<Record>::iterator it=recList.begin();
		for(int i=0;i<5;i++){
			it++;
		}
		for(int i=10;i<20;i++){
			it=recList.insert(data[i],it);
			add(data[i],5,mirror,i);
		}
		DList<int>::iterator it2=intList.begin();
		for(int i=0;i<cap;i++){
			it2=intList.insert(intData[cap-1-i],it2);
		}
		cout << "test 2: create list, insert some records." << endl;	
		if(!checkList(recList,mirror,20)){
			passtest=false;
			cout << "Error 2a: Bug in either constructor, push_front(), push_back(),insert() or iterator (operator++(int) or operator*)"<< endl;
			#if VERBOSE <= 1
			printLists(recList,mirror,20);
			#endif
		}
		if(!checkList(intList,intMirror,cap)){
			passtest=false;
			cout << "Error 2b: Bug in either constructor, insert()"<< endl;
			#if VERBOSE <= 1
			printLists(intList,intMirror,cap);
			#endif
		}
	}
	if(passtest){
		numPassed++;
		cout << "test 3: check iterator ++ and -- operators"  << endl;
		auto it=recList.begin();
		for(int i=0;i<19 && passtest;i++){
			if(mirror[i] != *(it++)){
				passtest=false;
				cout << "Error 3a: postfix ++ operator is not returning iterator to correct node" << endl;
			}
		}
		for(int i=19;i>0 && passtest;i--){
			if(mirror[i] != *(it--)){
				passtest=false;
				cout << "Error 3b: postfix -- operator is not returning iterator to correct node" << endl;
			}
		}
		for(int i=0;i<19 && passtest;i++){
			if(*(++it) != mirror[i+1]){
				passtest=false;
				cout << "Error 3c: prefix ++ operator is not returning iterator to correct node" << endl;
			}
		}
		for(int i=19;i>0 && passtest;i--){
			if(*(--it) != mirror[i-1]){
				passtest=false;
				cout << "Error 3d: prefix -- operator is not returning iterator to correct node" << endl;
			}
		}
	}


	DList<Record> recCopy=recList;
	DList<int> intCopy=intList;
	DList<Record> recCopy2=recCopy;
	DList<int> intCopy2=intCopy;
	duplicateArray(data,mirror,20);
	duplicateArray(intData,intMirror,cap);

	if(passtest){
		numPassed++;
		cout << "test 4: create a duplicate of the lists with copy constructor, ensure they match" << endl;
		if(!checkList(recList,mirror,20)){
			passtest=false;
			cout << "Error 4a: Bug in copy constructor"  << endl;  		
		}
		if(!checkList(intList,intMirror,cap)){
			passtest=false;
			cout << "Error 4b: Bug in copy constructor" << endl;
		}
	}
	if(passtest){
		numPassed++;
		cout << "test 5: test search() function" << endl;
		for(int i=0;i<20;i++){
			DList<Record>::iterator it = recList.search(mirror[i]);
			if(it == recList.end() || *it != mirror[i]){
				passtest=false;
				cout << "Error 5a: Bug in search, iterator returned does not point at correct node" << endl;
			}
		}
		for(int i=0;i<numSearches&& passtest;i++){
			int pick=rand()%cap;
			DList<int>::const_iterator it = intList.search(intMirror[pick]);
			if(it == intList.end() || *it != intMirror[pick]){
				passtest=false;
				cout << "Error 5b: Bug in search, iterator returned does not point at correct node" << endl;
			}
		}
	}

	if(passtest){
		numPassed++;
		cout << "test 6: test erase function"  << endl;
		DList<Record>::iterator it;
		it=setIterator(recList,19);
		recList.erase(it);
		removeItem(mirror,19,20);
		if(!checkList(recList,mirror,19)){
			passtest=false;
			cout << "Error 6a: Bug in erase.  erased last item" << endl;
			#if VERBOSE >= 1
			printLists(recList,mirror,19);
			#endif
		}
		recList.erase(recList.begin());
		removeItem(mirror,0,19);
		if(!checkList(recList,mirror,18)){
			passtest=false;
			cout << "Error 6b: Bug in remove. erased first item" << endl;
			#if VERBOSE >= 1
			printLists(recList,mirror,18);
			#endif
		}
		for(int i=0;i<3 && passtest;i++){
			int pick=rand()%(18-i);
			it=setIterator(recList,pick);
			recList.erase(it);
			removeItem(mirror,pick,18-i);
			if(!checkList(recList,mirror,18-i-1)){
				passtest=false;
				cout << "Error 6c:  Bug in erase." << endl;
			}
		}
		intList.erase(intList.begin());
		removeItem(intMirror,0,cap);
		if(!checkList(intList,intMirror,cap-1)){
			passtest=false;
			cout << "Error 6d: Bug in erase, removed first" << endl;
		}
		intList.erase(intList.begin());
		removeItem(intMirror,0,cap-1);
		if(!checkList(intList,intMirror,cap-2)){
			passtest=false;
			cout << "Error 6e: Bug in erase. removed first again" << endl;
		}
		DList<int>::iterator it2;
		it2=setIterator(intList,cap-3);
		intList.erase(it2);
		removeItem(intMirror,cap-3,cap-2);
		if(!checkList(intList,intMirror,cap-3)){
			passtest=false;
			cout << "Error 6f: Bug in erase.removed last" << endl;
		}
		it2=setIterator(intList,cap-4);
		intList.erase(it2);
		removeItem(intMirror,cap-4,cap-3);
		if(!checkList(intList,intMirror,cap-4)){
			passtest=false;
			cout << "Error 6g: Bug in erase. removed last again" << endl;
		}
		for(int i=0;i<100 && passtest;i++){
			int pick=rand()%(cap-4-i);
			it2=setIterator(intList,pick);
			intList.erase(it2);
			removeItem(intMirror,pick,cap-4-i);
			if(!checkList(intList,intMirror,cap-i-5)){
				passtest=false;
				cout << "Error 6h:  Bug in erase." << endl;
			}
		}
	}
	if(passtest){
		numPassed++;
		cout << "test 7: ensure that copied list were not altered (proper deep copy was made) " << endl;
		if(!checkList(recCopy,data,20)){
			passtest=false;
			cout << "Error 7a: Bug in copy constructor, deep copy not made?"  << endl;
			#if VERBOSE >= 1
			printLists(recCopy,data,20);
			#endif
		}
		if(!checkList(intCopy,intData,cap)){
			passtest=false;
			cout << "Error 7b: Bug in copy constructor, deep copy not made?" << endl;
			#if VERBOSE >= 2
			printLists(intCopy,intData,cap);
			#endif
		}
	}
	if(passtest){
		numPassed++;
		cout << "test 8: test assignment operator"  << endl;
		recCopy2 = recList;
		intCopy2 = intList;
		if(!checkList(recCopy2,mirror,15)){
			passtest=false;
			cout << "Error 8a: Bug in assignment operator"  << endl;  		
			#if VERBOSE >= 1
			printLists(recCopy2,mirror,15);
			#endif
		}
		if(!checkList(intCopy2, intMirror,cap-104)){
			passtest=false;
			cout << "Error 8b: Bug in assignment operator"  << endl;  		
			#if VERBOSE >= 2
			printLists(intCopy2,intMirror,cap-104);
			#endif
		}
	}

	if(passtest){

		numPassed++;

		cout << "test 9: test assignment operator (deep copy)"  << endl;
		recCopy2.erase(recCopy2.begin());
		intCopy2.erase(intCopy2.begin());
		if(!checkList(recList,mirror,15)|| recList.empty() || recList.size()!=15){
			passtest=false;
			cout << "Error 9a: Bug in = operator, no deepcopy?"  << endl;
		}
		if(!checkList(intList,intMirror,cap-104)|| intList.empty() || intList.size()!=cap-104){
			passtest=false;
			cout << "Error 9b: Bug in = operator,no deep copy?"  << endl;     
		}
	}

	if(passtest){
		numPassed++;
		cout << "test 10: search for removed items" << endl;
		int pick[2]={0,19};
		for(int i=0;i<2;i++){
			auto it =recList.search(data[pick[i]]);
			if(it!=recList.end()){
				cout << "Error 10a: Bug in search, returned iterator is not correct" << endl;
				passtest=false;
			}
		}

	}
	if(passtest){
		int pick[4]={0,1,cap-1,cap-2};
		for(int i=0;i<4;i++){
			auto it = intList.search(intData[pick[i]]);
			if(it != intList.end()){

				passtest=false;
				cout << "Error 10b: Bug in search."<< endl;
			}
		}
	}
	if(passtest){
		numPassed++;
		cout << "test 11: test erase(front,back) function"  << endl;
		DList<Record>::iterator it;
		DList<Record>::iterator it2;
		Record tmp[20];
		duplicateArray(tmp,mirror,15);
		removeItem(tmp,0,15);
		it=setIterator(recCopy2,2);
		it2=setIterator(recCopy2,5);

		recCopy2.erase(it,it2);
		removeItem(tmp,4,14);
		removeItem(tmp,3,13);  
		removeItem(tmp,2,12);
		if(!checkList(recCopy2,tmp,11)){
			passtest=false;
			cout << "Error 11a: Bug erase(it) function"  << endl;  		
			#if VERBOSE >= 1
			printLists(recCopy2,tmp,11);
			#endif
		}
		if(!checkList(recList,mirror,15)){
			passtest=false;
			cout << "Error 11b: assignment operator, (deep copy not made?)"  << endl;  		
			#if VERBOSE >= 1
			printLists(recList,mirror,15);
			#endif

		}
	}
	if(passtest){
		numPassed++;
		cout << "test 12: test erase(it,end()) function"  << endl;
		DList<Record>::iterator it;
		DList<Record>::iterator it2;
		Record tmp[20];
		recCopy2=recList;
		intCopy2=intList;
		duplicateArray(tmp,mirror,15);
		it=setIterator(recCopy2,11);

		recCopy2.erase(it,recCopy2.end());
		removeItem(tmp,14,15);
		removeItem(tmp,13,14);  
		removeItem(tmp,12,13);
		removeItem(tmp,11,12);
		if(!checkList(recCopy2,tmp,11)){
			passtest=false;
			cout << "Error 12a: Bug erase(first,last) function"  << endl;  		
			#if VERBOSE >= 1
			printLists(recCopy2,tmp,11);
			#endif
		}
		if(!checkList(recList,mirror,15)){
			passtest=false;
			cout << "Error 12b: assignment operator, (deep copy not made?)"  << endl;  		
			#if VERBOSE >= 1
			printLists(recList,mirror,15);
			#endif

		}
	}
	DList<Record> recCopy3=std::move(recList);
	if(passtest){
		numPassed++;
		cout << "test 13: test move constructor"  << endl;

		if(!checkList(recCopy3,mirror,15)){
			passtest=false;
			cout << "Error 13a: Bug in move constructor"  << endl;  		
			#if VERBOSE >= 1
			printLists(recCopy3,mirror,15);
			#endif
		}
		if(recList.begin() == recCopy3.begin()){
			passtest=false;
			cout << "Error 13b: Bug in move constructor"  << endl;
			cout << "original and copy both refer to same list" << endl; 		
		}
	}

	if(passtest){
		numPassed++;
		recList=recCopy3;
		cout << "test 14: test move assignment operator"  << endl;
		recCopy3=std::move(recList);
		if(!checkList(recCopy3,mirror,15)){
			passtest=false;
			cout << "Error 14a: Bug in move assignment operator"  << endl;  		
			#if VERBOSE >= 1
			printLists(recCopy3,mirror,15);
			#endif
		}
		if(recList.begin() == recCopy3.begin()){
			passtest=false;
			cout << "Error 14b: Bug in move assignment operator"  << endl;
			cout << "original and copy both refer to same list" << endl;  		
		}
	}


	if(passtest){
		numPassed++;
	}
	if(numPassed == 14){
		cout << "Testing for Assignment 1, part 1 completed successfully" << endl;
		cout << "14/14 tests passed" << endl;
	}
	else{
		cout << numPassed << " / 14 tests passed.  Looks like you still" << endl;
		cout << "have some work to do" << endl;
	}
	return 0;
}
template <typename T>
void addFront(const T& data,T arr[],int sz){
	add(data,0,arr,sz);
}
template <typename T>
void addBack(const T& data,T arr[],int sz){
	arr[sz]=data;
}
template <typename T>
void add(const T& data,int idx, T arr[],int sz){
	for(int i=sz-1;i>=idx;i--){
		arr[i+1]=arr[i];
	}
	arr[idx]=data;
}
template <typename T>
void removeItem(T arr[],int idx,int sz){
	for(int i=idx;i<sz-1;i++){
		arr[i]=arr[i+1];
	}
}
template <typename T>
bool checkList(const DList<T>& list,const T array[],int sz){
	bool rc=true;
	auto it=list.begin();
	int i;
	for(i=0;i<sz && it!=list.end() && rc;i++,it++){
		if(*it != array[i]){
			rc=false;
		}
	}
	if(it!=list.end() || i < sz){
		rc=false;
	}
	return rc;
}

template <typename T>
void duplicateArray(T dest[], const T src[],int sz){
	for(int i=0;i<sz;i++){
		dest[i]=src[i];
	}
}

template <typename T>
void printLists(const DList<T>& list,const T array[],int sz){
	auto it=list.begin();
	for(int i=0;i<sz;i++,it++){
		cout << "correct: "  << array[i] << " your list: " << *it << endl;
	}
}
//returns iterator to the idx'th node in the list
template <typename T>
typename DList<T>::iterator setIterator(DList<T>& list, int idx){
	typename DList<T>::iterator it=list.begin();
	for(int i=0;i<idx;i++){
		it++;
	}
	return it;
}