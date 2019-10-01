DROP DATABASE IF EXISTS LOG;
CREATE DATABASE LOG;
USE LOG;
CREATE TABLE Transactions (id INTEGER(5), student_id INTEGER(5), name_ VARCHAR (200), date_ DATE, time_ TIME);
CREATE TABLE Suspicious (path_to_img VARCHAR(200), date_ DATE, time_ TIME);