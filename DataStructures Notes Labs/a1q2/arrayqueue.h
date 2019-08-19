template<class T>
class Queue{

	void grow(){
		T* tmp=new T[capacity_*2];
		int j=front_;
		for(int i=0;i<used_;i++){
			tmp[i]=data[j];
			j=(j+1)%capacity_;
		}
		delete [] data_;
		data_=tmp;
		capacity_=capacity_*2;
		front_=0;
		back_=used_;
	}
	T* data_;
	int capacity_;
	int used_;
	int front_;  //index of front of list
	int back_;	 //index of where to put 
	             //new things
public:
	Queue(){
		capacity_=50;
		data_=new T[capacity_];
		used_=0;
		front_=0;
		back_=0;
	}
	void enqueue(const T& data){
		data_[back_]=data;
		back_=(back_+1)%capacity_;
/*		if(back_==capacity_-1){
			back_=0;
		}
		else{
			back_++;
		}*/
	}
	void dequeue(){
		front_=(front_+1)% capacity_;
	}
	T front() const{
		return data_[front_];	
	}
	bool empty(){
	}

};