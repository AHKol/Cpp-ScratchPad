template <typename T>
class DList{

	struct Node{
		T data_;
		Node* next_;
		Node* prev_;
		Node(const T& d=T{}, Node* n=nullptr, Node* p=nullptr){
			data_=d;
			next_=n;
			prev_=p;
		}
	};
	Node* front_;
	Node* back_;
public:
	DList(){
		front_= new Node();
		back_=new Node();
		front_->next_ = back_;
		back_->prev_=front_;
	}
	//puts a new node containing data between the front sentinel (first node)
	//and second node (could be data, could be back sentinel)
	void push_front(const T& data){
		Node* first=front_;
		Node* second=first->next_;
		//make a new node with data ,next pointer points to second
		//node, previous is sentinel
		Node* nn=new Node(data,second,first);
		first->next_=nn;
		second->prev_=nn;
	}
	//puts a new node containing data at the back of the linked list
	void push_back(const T& data);
	//removes the first node in linked list
	void pop_front(){
		//if list is not empty of data
		if(front_->next_ != back_){
			Node* rm = front_->next_;
			Node* rmNext=rm->next_;
			front_->next_=rmNext;
			rmNext->prev_=front_;
			delete rm;
		}
	}
	//removes last node in linked list
	void pop_back();
	void print() const;
	~DList();
	const_iterator begin() const{return const_iterator(front_->next_);}
	const_iterator end() const{return const_iterator(back_);}
	class const_iterator{
		friend class DList;
		Node* curr_;
		//assign curr_ with n;
		const_iterator(Node* n){
			curr_=n;
		}
	public:
		const_iterator(){
			curr_=nullptr;
		}
		const_iterator operator++(){
			//++x
			curr_=curr_->next_;
			return *this;
		}
		const_iterator operator++(int){
			//x++
			const_iterator old=*this;
			curr_=curr_->next_;
			return old;
		}
		const_iterator operator--(){
			//--x
			curr_=curr_->prev_;
			return *this;
		}
		const_iterator operator--(int){
			//x--
			const_iterator old=*this;
			curr_=curr_->prev_;
			return old;
		}
		const T& operator*() const{
			return curr_->data_;
		}
		bool operator!=(const_iterator rhs) const{
			return (curr_!=rhs.curr_);
		}
		bool operator==(const_iterator rhs) const{
			return (curr_==rhs.curr_);
		}
	};	



};