template <typename T>
class DList {
	//size of DList allows for O(1) size() function
	int size_;
	struct Node {
		T data_;
		Node* next_;
		Node* prev_;
		Node(const T& data = T{}, Node* nx = nullptr, Node* pr = nullptr) { //node constructor with defaults
			data_ = data;
			next_ = nx;
			prev_ = pr;
		}
	};
	Node* front_;
	Node* back_;

public:
	class const_iterator {
	protected:
		//iterator tracks what it's pointing to with position_
		Node * position_;
		friend class DList;
	public:
		const_iterator() {	//default constructor
			position_ = nullptr;
		}
		const_iterator(Node* position) {	//iterator constructed pointing to position node
			position_ = position;
		}
		const_iterator operator++() {	//change position of iterator to next node
			position_ = position_->next_;
			return *this;
		}
		const_iterator operator++(int) {	//change position of iterator to next node after semicolon 
			const_iterator ret(this->position_);
			position_ = position_->next_;
			return ret;
		}
		const_iterator operator--() {	//change position of iterator to previous node
			position_ = position_->prev_;
			return *this;
		}
		const_iterator operator--(int) {	//change position of iterator to previous node after semicolon
			const_iterator ret(position_);
			position_ = position_->prev_;
			return ret;
		}
		bool operator==(const_iterator rhs) {	//compare nodes within iterators
			if (position_ == rhs.position_)
				return true;					//return true if iterators point to same node
			else
				return false;
		}
		bool operator!=(const_iterator rhs) {	//compare nodes within iterators
			if (position_ == rhs.position_)
				return false;
			else
				return true;					//return true if iterators do no point to same node
		}
		const T& operator*()const {				
			return position_->data_;			//return reference to data within iterator
		}
	};
	class iterator :public const_iterator {
	public:
		iterator() {	//default constructor
			const_iterator::position_ = nullptr;
		}
		iterator(Node* position) {	//iterator constructed pointing to position node
			const_iterator::position_ = position;
		}
		iterator operator++() {	//change position of iterator to next node
			const_iterator::position_ = const_iterator::position_->next_;
			return const_iterator::position_;
		}
		iterator operator++(int) {	//change position of iterator to next node after semicolon
			iterator ret(this->position_);
			const_iterator::position_ = const_iterator::position_->next_;
			return ret;
		}
		iterator operator--() {	//change position of iterator to previous node
			const_iterator::position_ = const_iterator::position_->prev_;
			return const_iterator::position_;
		}
		iterator operator--(int) {	//change position of iterator to previous node after semicolon
			iterator ret(this->position_);
			const_iterator::position_ = const_iterator::position_->prev_;
			return ret;
		}
		T& operator*() {	//return reference to within iterator
			return this->position_->data_;
		}
		const T& operator*()const {	//return reference to data within iterator
			return this->position_->data_;
		}
	};
	DList();
	~DList();
	DList(const DList& rhs);
	DList& operator=(const DList& rhs);
	DList(DList&& rhs);
	DList& operator=(DList&& rhs);
	iterator begin() {
		if (front_ != nullptr) {	//return an iterator pointing to first node with data
			return iterator(front_->next_);
		}
		else {
			return nullptr;
		}
	}
	iterator end() {	//return an iterator pointing to node after last node with data
		return iterator(back_);
	}
	const_iterator begin() const {	//return an iterator pointing to first node with data
		return const_iterator(front_->next_);
	}
	const_iterator end() const {	//return an iterator pointing to node after last node with data
		return const_iterator(back_);
	}
	void push_front(const T& data);
	void push_back(const T& data);
	void pop_front();
	void pop_back();
	iterator insert(const T& data, iterator it);
	iterator search(const T& data);
	const_iterator search(const T& data) const;
	iterator erase(iterator it);
	iterator erase(iterator first, iterator last);
	bool empty() const;
	int size() const;
};

template <typename T>
DList<T>::DList() {	//construct empty DList
	size_ = 0;
	front_ = new Node();
	back_ = new Node();
	front_->next_ = back_;
	back_->prev_ = front_;
}
template <typename T>
DList<T>::~DList() {	//destroy DList and any data within it
	Node* curr = front_;
	while (curr) {
		Node* rm = curr;
		curr = curr->next_;
		delete rm;
	}
}
template <typename T>
DList<T>::DList(const DList& rhs) {	//copy constructor
	size_ = 0;
	front_ = new Node();
	back_ = new Node();
	front_->next_ = back_;
	back_->prev_ = front_;
	const_iterator it = rhs.begin();
	do {	
		push_back(*it);
		it++;
	} while (it != rhs.end());	//loop through rhs and copy each node
}
template <typename T>
DList<T>& DList<T>::operator=(const DList& rhs) {	//assignment operator
	if (this == &rhs) return *this;
	DList temp(rhs);
	swap(*this, temp);
	return *this;
}
template <typename T>
DList<T>::DList(DList&& rhs) {	//move constructor
	this->front_ = move(rhs.front_);
	this->back_ = move(rhs.back_);
	this->size_ = move(rhs.size_);
	rhs.front_ = nullptr;
	rhs.back_ = nullptr;
	rhs.size_ = 0;
}

template <typename T>
DList<T>& DList<T>::operator=(DList&& rhs) {	//move assignment operator
	if (this == &rhs) return *this;
	this->front_ = move(rhs.front_);
	this->back_ = move(rhs.back_);
	this->size_ = move(rhs.size_);
	rhs.front_ = nullptr;
	rhs.back_ = nullptr;
	rhs.size_ = 0;
	return *this;	
}

template<typename T>
inline void DList<T>::push_front(const T& data) {	//place new node in front of list
	size_++;	//increment list size tracker
	Node* first = front_;
	Node* second = first->next_;	//simplify tracking first 2 nodes
	Node* nn = new Node(data, second, first);	//construct new node with data and connections to first and second
	first->next_ = nn;
	second->prev_ = nn;	//tie first and second to new node
}
template<typename T>
inline void DList<T>::push_back(const T& data) {	//place new node in back of list
	size_++;	//increment list size tracker
	Node* first_last = back_; // simplify tracking last 2 nodes
	Node* second_last = back_->prev_;
	Node* nn = new Node(data, first_last, second_last);	//construct new node with data and connections to first and second
	first_last->prev_ = nn;
	second_last->next_ = nn;	//tie first and second to new node
}
template <typename T>
void DList<T>::pop_front() {	//if list is not empty, remove first node
	if (front_->next_ != back_) {
		size_--;	//dencrement list size tracker
		Node* rm = front_->next_;	//track remove node
		Node* rmNext = rm->next_;	//track next node
		front_->next_ = rmNext;		//move front to next node
		rmNext->prev_ = front_;		//move prev to sentinel node
		delete rm;					//delete node
	}
}
template <typename T>
void DList<T>::pop_back() {	//if list is not empty, remove last node
	if (front_->next_ != back_) {	
		size_--;	//dencrement list size tracker
		Node* rm = back_->prev_;	//track remove node
		Node* rmPrev = rm->prev_;	//track next node
		back_->prev_ = rmPrev;		//move back to next node
		rmPrev->next_ = back_;		//move prev to sentinel node
		delete rm;					//delete node
	}
}

template <typename T>
typename DList<T>::iterator DList<T>::insert(const T& data, iterator it) {	//place new node where iterator is sitting, returns an iterator at new node
	Node* next = it.position_;
	Node* prev = it.position_->prev_;		//track 2 neighboring nodes
	Node* nn = new Node(data, next, prev);	//create new node with conections to neighbors
	prev->next_ = nn;	
	next->prev_ = nn;	//tie neighbors to new node
	size_++;	//increment size tracker
	return iterator(nn);	
}

template <typename T>
typename DList<T>::iterator DList<T>::search(const T& data) {	//returns iterator containing pointer to matching data
	iterator it = begin();	
	do {
		if (*it == data) {	//loop to end of array
			return it;		//return iterator if finds a match
		}
		it++;
	} while (it != end());
	return end();			//if no match, return node after last
}

template <typename T>
typename DList<T>::const_iterator DList<T>::search(const T& data) const {	//returns iterator containing
	iterator it = begin();
	do {				
		if (*it == data) {	//loop to end of array
			return it;		//return iterator if finds a match
		}
		it++;
	} while (it != end());
	return end();			//if no match, return node after last
}

template <typename T>
typename DList<T>::iterator DList<T>::erase(iterator it) {	//remove node that iterator is pointing to, return iterator at next node
	Node* rm = it.position_;	
	Node* prev = it.position_->prev_;
	Node* next = it.position_->next_;	//track remove node and node neighbors

	prev->next_ = next;
	next->prev_ = prev;		//tie node neighbors together

	delete rm;				//remove node			
	size_--;				//decrement tracker
	return iterator(next);
}

template <typename T>
typename DList<T>::iterator DList<T>::erase(iterator first, iterator last) {	//remove nodes between the 2 iterators, excluding node in last iterator, return iterator at next node
	Node* position = first.position_;		
	while (position != last.position_) {	//remove first node and tie the others together, loop until first node matches last and return
		Node* rm = position;
		Node* prev = position->prev_;
		Node* next = position->next_;

		position = next;

		prev->next_ = next;
		next->prev_ = prev;
		size_--;
		delete rm;
	}
	return last;
}
template <typename T>
bool DList<T>::empty() const {	//check if last sentinel is after first
	if (front_->next_ == back_)
		return true;			//return true if it is
	else
		return false;
}
template <typename T>
int DList<T>::size() const {
	return size_;			//return a size_ int that is incremented and decremented through link list's life
}
