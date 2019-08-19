#ifndef A3_H
#define A3_H
#include <string>
using namespace std;
class SpellCheck{

  struct Node{
    bool isEnd_; //used to determine if it is the end node belonging to a string of characters
    char letter_; //used to store the letter(or children_ of a string)
    Node* children_[26]; //used to store the next node

    Node() {
      isEnd_ = false;
      letter_ = '\0';
      for (int i = 0; i < 26; i++){
        children_[i] = nullptr;
      }
    }
  };

Node* root_;


public:

	SpellCheck(string data[],int sz);
	void addWord(const string& newWord);
  void addWord(const string& newWord, int pos, Node* root);
	bool lookup(const string& word) const;
  bool lookup(const string& word, int pos, Node* root) const;
	int suggest(const string& partialWord, string suggestions[]) const;
	~SpellCheck();
};
#endif
