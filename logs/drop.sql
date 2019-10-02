/* "drop.sql"
   Drops LOG database and creates an empty one.
   */
DROP DATABASE IF EXISTS LOG;
CREATE DATABASE LOG;
USE LOG;
CREATE TABLE Transactions (student_id INTEGER(5), student_name VARCHAR (200), date DATE, time TIME);
CREATE TABLE Suspicious (path_to_img VARCHAR(200), date DATE, time TIME);