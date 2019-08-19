#include "a3.h"
#include <string>
#include <iostream>
using namespace std;

//use a macro to get the value of a index for a character.
#define CHAR_TO_INDEX(c) ((int)c - (int)'a')

SpellCheck::SpellCheck(string data[],int sz){
  root_ = new Node();

  for (int i = 0; i < sz; i++){
    addWord(data[i]);
  }
}
void SpellCheck::addWord(const string& newWord){
  addWord(newWord, 0, root_);
}
/*This function recurses through a string, traversing through the tree & creates new character Nodes when necessary
pos is the index value of the string, root is the position in the tree we are currently at */
void SpellCheck::addWord(const string& newWord, int pos, Node* root){
  //base case
  //std::cout << newWord.length() << std::endl;
  if (newWord.length() == pos){
    //std::cout << "End of word" << std::endl;
    root->isEnd_=true;
    return;
  }
  if(root->children_[CHAR_TO_INDEX(newWord.front())] == nullptr){
    Node* nn = new Node();
    nn->letter_ = newWord[pos];
    root = root->children_[CHAR_TO_INDEX(newWord.front())] = nn; //traverse root to it's child, and set the child to the new node.
    //std::cout << "Creating new node " << nn->letter_ << std::endl;
    addWord(newWord, pos+1, root);
  }
  else {
    addWord(newWord, pos+1, root->children_[CHAR_TO_INDEX(newWord.front())]);
  }
}

bool SpellCheck::lookup(const string& word) const{
  return lookup(word, 0, root_);
}

/*This function will recurse through the word passed, checking that the letter_
stored in each node matches the position in the word. */
bool SpellCheck::lookup(const string& word, int pos, Node* root) const{
  //base case, we successfully traverse through the entire word.
  if(word.length()  == pos){
    std::cout << "Found" << std::endl;
    return true;
  }
  if(root->children_[CHAR_TO_INDEX(word.front())] != nullptr && root->children_[CHAR_TO_INDEX(word.front())]->letter_ == word[pos]){
    lookup(word, pos+1, root->children_[CHAR_TO_INDEX(word.front())]);
  }
  else{
    return false;
  }

}

int SpellCheck::suggest(const string& partialWord, string suggestions[]) const{


}
SpellCheck::~SpellCheck(){

}


/*int main(void) {
  string list[4]{ "dog", "cat", "bird", "birds"};
  SpellCheck test(list, 4);
  std::cout << test.lookup("birds") << std::endl;
  std::cout << test.lookup("bird") << std::endl;
  std:: cout << test.lookup("eeeek") << std::endl;
  return 0;
} */
