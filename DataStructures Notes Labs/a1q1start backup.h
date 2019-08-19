
template <typename T>

class DList{
	int size_;
	struct Node{
		T data_;
		Node* next_;
		Node* prev_;
		Node(const T& data=T{},Node* nx=nullptr,Node* pr=nullptr){
			data_ = data;
			next_ = next;
			prev_ = prev;
		}
	};
	Node* front_;
	Node* back_;

public:
	class const_iterator{
	public:
		Node* position_;
		const_iterator(){
			position_ = nullptr;
		}
		const_iterator operator++(){
			position_ = position_.next_;
			return position_;
		}
		const_iterator operator++(int count){
			for (int i = 0; i < size(); i++) {
				position_ = position_.next_;
			}
			return position_;
		}
		const_iterator operator--(){
			position_ = position_.prev_;
			return position_;
		}
		const_iterator operator--(int){
			for (int i = 0; i < size(); i++) {
				position_ = position_.prev_;
			}
			return position_;
		}
		bool operator==(const_iterator rhs){
			if (position_ == rhs.position_)
				return true;
			else
				return false;
		}
		bool operator!=(const_iterator rhs){
			if (position_ == rhs.position_)
				return false;
			else
				return true;
		}
		const T& operator*()const{
			return position_->data_;
		}
	};
	class iterator:public const_iterator{
	public:
		iterator(){
		//parent constructs nullptr as position
		}
		iterator operator++(){
			position_ = position_.next_;
			return position_;
		}
		iterator operator++(int){
			for (int i = 0; i < size(); i++) {
				position_ = position_.next_;
			}
			return position_;
		}
		iterator operator--(){
			position_ = position_.prev_;
			return position_;
		}
		iterator operator--(int){
			for (int i = 0; i < size(); i++) {
				position_ = position_.prev_;
			}
			return position_;
		}
		T& operator*(){
			//todo current compiler error
			return this->position_->data_;
		}
		const T& operator*()const{
			return position_.data_;
		}
	};
	DList();
	~DList();
	DList(const DList& rhs);
	DList& operator=(const DList& rhs);
	DList(DList&& rhs);
	DList& operator=(DList&& rhs);
	iterator begin(){
		it_.position_ = front_->next_;
		return it_;
	}
	iterator end(){
		return iterator(front_);
	}
	const_iterator begin() const{
		const_iterator::position_ = front_.next_;
		return *this;
	}
	const_iterator end() const{
		iterator::position_ = back_;
		return *this;
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
DList<T>::DList(){
	front_ = new Node();
	back_ = new Node();
	front_->next_ = back_;
	back_->prev_ = front_;
}
template <typename T>
DList<T>::~DList(){
	Node* curr = front_;
	while (curr) {
		Node* rm = curr;
		curr = curr->next_;
		delete rm;
	}
}
template <typename T>
DList<T>::DList(const DList& rhs){
	front_ = rhs.front_;
	back_ = rhs.back_;
	size_ = rhs.size_;
	//todo
	//do deep copy
}
template <typename T>
DList<T>& DList<T>::operator=(const DList& rhs){
	front_ = rhs.front_;
	back_ = rhs.back_;
	size_ = rhs.size_;
	//todo
	//do deep copy
}
template <typename T>
DList<T>::DList(DList&& rhs){
	front_ = rhs.front_;
	back_ = rhs.back_;
	size_ = rhs.size_;
	delete rhs;
}

template <typename T>
DList<T>& DList<T>::operator=(DList&& rhs){
	front_ = rhs.front_;
	back_ = rhs.back_;
	size_ = rhs.size_;
	delete rhs;
}

template<typename T>
inline void DList<T>::push_front(const T& data) {
	Node* first = front_;
	Node* second = first->next_;
	Node* nn = new Node(data, second, first);
	first->next_ = nn;
	second->prev_ = nn;
}
template<typename T>
inline void DList<T>::push_back(const T& data) {
	Node* first_last = back_;
	Node* second_last = back_->prev_;
	Node* nn = new Node(data, first_last, second_last);
	first_last->prev_ = nn;
	second_last->next_ = nn;
}
template <typename T>
void DList<T>::pop_front(){
	if (front_->next_ != back_) {
		Node* rm = front_->next_;
		Node* rmNext = rm->next_;
		front_->next_ = rmNext;
		rmNext->prev_ = front_;
		delete rm;
	}
}
template <typename T>
void DList<T>::pop_back(){
	if (front_->next_ != back_) {
		Node* rm = back_->prev_;
		Node* rmPrev = rm->prev_;
		back_->prev_ = rmPrev;
		rmPrev->next_ = back_;
		delete rm;
	}
}

template <typename T>
typename DList<T>::iterator DList<T>::insert(const T& data, iterator it){
	Node* next = it;
	Node* prev = it->prev_;
	Node nn(data, next, prev);
	prev->next_ = nn;
	next->prev_ = nn;
}

template <typename T>
typename DList<T>::iterator DList<T>::search(const T& data){
	Node* position = front_;
	for (int i = 0; i < size() i++) {
		if (position->data_ == data) {
			return position;
		}
	}
	return end();
}

template <typename T>
typename DList<T>::const_iterator DList<T>::search(const T& data) const{
	Node* position = front_;
	for (int i = 0; i < size(); i++) {
		if (position->data_ == data) {
			return position;
		}
	}
	return end();
}

template <typename T>
typename DList<T>::iterator DList<T>::erase(iterator it){
	Node* rm = it;
	Node* prev = it->prev_;
	Node* next = it->next_;
	
	prev.next_ = next;
	next.prev_ = prev_;

	delete rm;

	return next;
}

template <typename T>
typename DList<T>::iterator DList<T>::erase(iterator first, iterator last){
	Node* position = first;
	while (position != last) {
		Node* rm = position;
		Node* ret = position->next_;
		Node* prev = position->prev_;
		Node* next = position->next_;

		position = next;

		prev.next_ = next;
		next.prev_ = prev;

		delete rm;
	}
	return last;
}
template <typename T>
bool DList<T>::empty() const{
	if (front_ == back_)
		return true;
	else
		return false
}
template <typename T>
int DList<T>::size() const{
	//todo retrofit size into everything
	return size_;
}
