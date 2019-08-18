#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "Util.h"

using namespace std;

class Student {
	int studentID;
	string studentName;
	map<int, vector < double>> gradeMap;
	double finalGrade;
public:
	Student() {};
	Student(int nStudentID, string nStudentName) {
		studentID = nStudentID;
		studentName = nStudentName;
	}
	int getID() {
		return studentID;
	}
	string getName() {
		return studentName;
	}
	double getFinalGrade() {
		double sum = 0;
		for (auto grade : gradeMap) {
			for (auto mark : grade.second) {
				sum += mark;
			}
		}
		return sum / gradeMap.size();
	}
	void addCourseMark(int courseID, double weightedGrade) {
		gradeMap[courseID].push_back(weightedGrade);
	}

	double getCourseGrade(int courseID) {
		if (gradeMap.find(courseID) == gradeMap.end()) {
			return -1.0;
		}

		double sum = 0;
		vector<double> marks = gradeMap.at(courseID);
		for (auto& mark : marks) {
			sum += mark;
		}
		return sum;
	}
};
struct Mark{
	int studentID;
	int testID;
	double mark;
};
struct Course{
	int courseID;
	string courseName;
	string teacherName;
};
struct Test {
	int testID;
	int courseID;
	double weight;
};
map<int, Student> CSVtoStudentMap(const vector<vector<string>> &studentsCSV) {
	map<int, Student> studentMap;

	//first elements are column names
	if ((studentsCSV[0][0] != "id") || (studentsCSV[0][1] != "name")) {
		return studentMap;
	}
	for (size_t i = 1; i < studentsCSV.size(); i++) {
		if (studentsCSV[i].size() != 2) {
			studentMap.clear();
			return studentMap;
		}
		int key;
		int studentID;
		try {
			studentID = stoi(studentsCSV[i][0], nullptr);
		}
		catch (...)
		{
			studentMap.clear();
			return studentMap;
		}
		key = studentID;
		string studentName = studentsCSV[i][1];
		studentMap.insert(make_pair(key, Student(studentID, studentName)));
	}

	return studentMap;
}
map<int, Course> CSVtoCourseMap(const vector<vector<string>> &courseCSV) {
	map<int, Course> courseMap;

	//first elements are column names
	if ((courseCSV[0][0] != "id") || (courseCSV[0][1] != "name") || courseCSV[0][2] != "teacher") {
		return courseMap;
	}
	for (size_t i = 1; i < courseCSV.size(); i++) {
		if (courseCSV[i].size() != 3) {
			courseMap.clear();
			return courseMap;
		}
		Course tempCourse;
		int key = stoi(courseCSV[i][0], nullptr);
		try {
			tempCourse.courseID = stoi(courseCSV[i][0], nullptr);
		}
		catch (...)
		{
			courseMap.clear();
			return courseMap;
		}
		tempCourse.courseName = courseCSV[i][1];
		tempCourse.teacherName = courseCSV[i][2];

		courseMap.insert(make_pair(key, tempCourse));
	}
	return courseMap;
}
//test_ID, Student_ID, Mark
map<int, map<int, Mark>> CSVtoMarkMap(const vector<vector<string>>& markCSV) {
	map<int, map<int, Mark>> markMap;
	//first elements are column names
	if ((markCSV[0][0] != "test_id") || (markCSV[0][1] != "student_id") || markCSV[0][2] != "mark") {
		return markMap;
	}
	for (size_t i = 1; i < markCSV.size(); i++) {
		if (markCSV[i].size() != 3) {
			markMap.clear();
			return markMap;
		}
		Mark tempMark;
		try {
			tempMark.testID = stoi(markCSV[i][0]);
			tempMark.studentID = stoi(markCSV[i][1]);
			tempMark.mark = stod(markCSV[i][2]);
		}
		catch (...)
		{
			markMap.clear();
			return markMap;
		}
		markMap[tempMark.studentID].insert(make_pair(tempMark.testID, tempMark));
	}
	return markMap;
}
map<int, Test> CSVtoTestMap(const vector<vector<string>> &testCSV) {
	map<int, Test> testMap;
	//first elements are column names
	if ((testCSV[0][0] != "id") || (testCSV[0][1] != "course_id") || testCSV[0][2] != "weight") {
		return testMap;
	}
	for (size_t i = 1; i < testCSV.size(); i++) {
		Test tempTest;
		if (testCSV[i].size() != 3) {
			testMap.clear();
			return testMap;
		}
		try {	
			tempTest.testID = stoi(testCSV[i][0]);
			tempTest.courseID = stoi(testCSV[i][1]);
			tempTest.weight = stod(testCSV[i][2]);
		}
		catch (...)
		{
			testMap.clear();
			return testMap;
		}
		testMap.insert(make_pair(tempTest.testID, tempTest));
	}
	return testMap;
}
int main() {

	vector<vector<string>> studentsCSV;
	vector<vector<string>> marksCSV;
	vector<vector<string>> testsCSV;
	vector<vector<string>> coursesCSV;

	//read CSVs
	try {
		csvRead("students.csv", ',', studentsCSV);
		csvRead("marks.csv", ',', marksCSV);
		csvRead("tests.csv", ',', testsCSV);
		csvRead("courses.csv", ',', coursesCSV);
	}
	catch (string e) {
		cerr << e << endl;
		return 1;
	}

	MapCSV CSVstudentClass(studentsCSV, "dc");

	//Verify and Map data
	map<int, Student> studentMap = CSVtoStudentMap(studentsCSV);
	map<int, map<int, Mark>> markMap = CSVtoMarkMap(marksCSV);
	map<int, Test> testMap = CSVtoTestMap(testsCSV);
	map<int, Course> courseMap = CSVtoCourseMap(coursesCSV);

	//if any are size 0 then read has failed.
	if (studentMap.size() == 0) {
		cerr << "Students.csv improper file format" << endl;
		return 3;
	}
	if (markMap.size() == 0) {
		cerr << "Marks.csv improper file format" << endl;
		return 3;
	}
	if (testMap.size() == 0) {
		cerr << "Tests.csv improper file format" << endl;
		return 3;
	}
	if (courseMap.size() == 0) {
		cerr << "Courses.csv improper file format" << endl;
		return 3;
	}

	//populate students with course information
	for (auto& student : studentMap) {
		//get student's marks
		map<int, Mark> studentMarks = markMap.at(student.second.getID());
		for (auto& studentMark : studentMarks) {
			//add weighted mark to student's course
			int testId = studentMark.second.testID;
			int courseId = testMap.at(testId).courseID;
			double tempMark = studentMark.second.mark;
			
			tempMark *= testMap.at(testId).weight / 100;
			student.second.addCourseMark(courseId, tempMark);
		}

	}

	//print output
	ofstream fout;
	fout.open("GradeOutput.txt", ios::trunc);
	for (auto& student : studentMap) {
		fout << "Student Id : " << student.second.getID() << ", name : " << student.second.getName() << endl;
		fout.precision(2);
		fout << "Total Average : " << fixed << student.second.getFinalGrade() << "%" << endl << endl;

		//for possible courses
		for (auto& course : courseMap) {
			double courseTotal = student.second.getCourseGrade(course.second.courseID);
			//course is not associated with student
			if (courseTotal < 0) {
				continue;
			}
			fout << '\t' << "Course: " << course.second.courseName << ", Teacher : " << course.second.teacherName << endl;
			fout << '\t' << "Final Grade : " << courseTotal << "%" << endl << endl;
		}
		fout << endl << endl;
	}


	return 0;
}