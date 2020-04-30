/* "drop.sql"
   Drops 'Log' database and creates an empty one.
   */
DROP DATABASE IF EXISTS Log;
CREATE DATABASE Log;
USE Log;
CREATE TABLE Activity (id VARCHAR(5), name VARCHAR(200), date DATE, time TIME);
/* student_id is VARCHAR(5) for cases like 00001-- can't be represented in python */
CREATE TABLE Unknown (path_to_img VARCHAR(200), date DATE, time TIME);
