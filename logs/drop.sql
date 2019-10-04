/* "drop.sql"
   Drops LOG database and creates an empty one.
   */
DROP DATABASE IF EXISTS LOG;
CREATE DATABASE LOG;
USE LOG;
CREATE TABLE Activity (student_id VARCHAR(5), student_name VARCHAR(200), date DATE, time TIME);
/* student_id is VARCHAR(5) for cases like 00001-- can't be represented in python */
CREATE TABLE Unknown (path_to_img VARCHAR(200), date DATE, time TIME);