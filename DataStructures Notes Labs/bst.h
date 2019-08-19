#include <arrayqueue>
template <typename T>
class BST{

	struct Node{
		T data_;     //data stored
		Node* left_; //ptr to left child
		Node* right_;//ptr to right childe
		Node(const T& data=T{},Node* left=nullptr,Node* right=nullptr){
			data_=data;
			left_=left;
			right_=right;
		}
	};
	Node* root_;
	void insert(const T& data,Node*& subtreeroot){
		if(subtreeroot == nullptr){
			//tree is empty
			subtreeroot=new Node(data);
		}
		else{
			if(data > subtreeroot->data_){
				insert(data,subtreeroot->right_);
			}
			else{
				insert(data,subtreeroot->left_);
			}
		}
	}

	void destory(Node* subroot){
		if(subRoot != nullptr){
			destory(subRoot->left_);
			destory(subRoot->right_);
			delete subroot;
		}
	}
public:
	BST(){
		//tree is empty
		root_=nullptr; 
	}
	~BST(){
		destroy(root_);
	}
	void insert(const T& data){
		insert(data, root_);
	}
	insertIterative(const T& data){
		Node* curr = root_;
		/*
		bool leftpoint
		while(curr != nullptr){
			if(data > curr->data_){
				curr = curr->right_;
			} else {
				curr = curr->left_;
			}
		}
		curr = new Node(data);
		*/
		if(curr == nullptr){
			root_=new Node(data);
		} else {
			bool inserted = false;
			while(!inserted){
				if(data > curr->data_){
					//belongs to left
					if (curr->left_ == nullptr){
						curr->left_ = new Node(data);
						inserted = true;
					} else {
						//move to new tree
						curr = curr->left_;
					}
				} else {
					//belongs to right
					if (curr->right_ == nullptr){
						curr->right_ = new Node(data);
						inserted = true;
					} else {
						//move to new tree
						curr = curr->right_;
					}
				}
			}
		}
	}
		void inOrderPrint(const Node* subroot){
		if(subRoot != nullptr){
			inOrderPrint(subRoot->left_);
			std::cout << subroot->data_ << std::endl;
			inOrderPrint(subRoot->right_);
		}
	}
	void preOrderPrint(const Node* subroot){
		if(subRoot != nullptr){	
			std::cout << subroot->data_ << std::endl;
			preOrderPrint(subRoot->left_);
			preOrderPrint(subRoot->right_);
		}
	}
	void postOrderPrint(const Node* subroot){
		if(subRoot != nullptr){
			postOrderPrint(subRoot->left_);
			postOrderPrint(subRoot->right_);
			std::cout << subroot->data_ << std::endl;
		}
	}
	void breadthFirstPrint(const Node* subroot){
		Queue<Node*> nodes;
		if(root_){
			nodes.enqueue(root_);
		}
		while(!nodes.empty()){
			Node* curr = nodes.front();
			if(curr->left_){
				nodes.enqueue(curr->left_);
			}
			if(curr->right_){
				nodes.enqueue(curr->right_);
			}
			std::cout << curr->data_ << std::endl;
			nodes.dequeue();
		}
	}
	void remove(T data, Node*& subroot){
		//find node
		//detach node <- hard part
		//delete node
		
		//if node has one child, make parent point to only child
		//if node has 2 children, we find the in order successor of that node and the inorder successor takes the place of the removed node
		//In order successor will be a leaf or it will have a right child
		
		//find
		if(subroot != nullptr){
			if(data == subroot->data_){ //if found data
				Node* rm = subroot;
				//detach
				if(!subroot->left_ && !subroot->right_){
					//if leaf node
					subroot = nullptr;
				} else if (subroot->left_ && !subroot->right_){
					//left child only
					subtRoot = subRoot->left_;
				} else if (!subroot->left_ && subroot->right_){
					//right child only
					subroot = subroot->right_;
				} else {
					//two children
				}
				//delete
				delete rm;
			}
			if (data < subroot->data_){
				remove(data, subroot->left_);
			} else {
				remove(data, subroot->right_);
			}
			
		}
	}
};