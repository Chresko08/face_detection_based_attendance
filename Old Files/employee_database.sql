CREATE TABLE employee(fname VARCHAR(20),mname VARCHAR(20),lname VARCHAR(20),ssn INT PRIMARY KEY,bdate DATE,address VARCHAR(50),sex CHAR(1),salary MEDIUMINT,super_ssn INT,FOREIGN KEY (super_ssn) REFERENCES employee(ssn),dno TINYINT);
CREATE TABLE department(dname VARCHAR(20),dnumber TINYINT PRIMARY KEY,mgr_ssn INT,FOREIGN KEY (mgr_ssn) REFERENCES employee(ssn),mgr_start_date DATE);
CREATE TABLE dept_locations(dnumber TINYINT,FOREIGN KEY (dnumber) REFERENCES department(dnumber),dlocation VARCHAR(20),PRIMARY KEY(dnumber,dlocation));
CREATE TABLE project(pname VARCHAR(20),pnumber TINYINT PRIMARY KEY,plocation VARCHAR(20),dnum TINYINT,FOREIGN KEY (dnum) REFERENCES department(dnumber));
CREATE TABLE works_on(essn INT,FOREIGN KEY (essn) REFERENCES employee(ssn),pno TINYINT,FOREIGN KEY (pno) REFERENCES project(pnumber),hours FLOAT,PRIMARY KEY(essn,pno));
CREATE TABLE dependent(essn INT,FOREIGN KEY (essn) REFERENCES employee(ssn),dependent_name VARCHAR(20),PRIMARY KEY(essn,dependent_name),sex CHAR(1),bdate DATE,relationship VARCHAR(20));
ALTER TABLE employee ADD FOREIGN KEY (dno) REFERENCES department(dnumber);


INSERT INTO employee VALUES("John","B","Smith",123456789,"1965-01-09","731 Fondren,Houston,TX","M",30000,NULL,NULL);
INSERT INTO employee VALUES("Franklin","T","Wong",333445555,"1955-12-08","638 Voss,Houston,TX","M",40000,NULL,NULL);
INSERT INTO employee VALUES("Alicia","J","Zelaya",999887777,"1968-01-19","3321 Castle,Spring,TX","F",25000,NULL,NULL);
INSERT INTO employee VALUES("Jennifer","S","Wallace",987654321,"1941-06-20","291 Berry,Bellaire,TX","F",43000,NULL,NULL);
INSERT INTO employee VALUES("Ramesh","K","Narayan",666884444,"1962-09-15","975 Fire Oak,Humble,TX","M",38000,NULL,NULL);
INSERT INTO employee VALUES("Joyce","A","English",453453453,"1972-07-31","5631 Rice,Houston,TX","F",25000,NULL,NULL);
INSERT INTO employee VALUES("Ahmad","V","Jabbar",987987987,"1969-03-29","980 Dallas,Houston,TX","M",25000,NULL,NULL);
INSERT INTO employee VALUES("James","E","Borg",888665555,"1937-11-10","450 Stone,Houston,TX","M",55000,NULL,NULL);

INSERT INTO department VALUES("Research",5,333445555,"1988-05-22");
INSERT INTO department VALUES("Administration",4,987654321,"1995-01-01");
INSERT INTO department VALUES("Headquarters",1,888665555,"1981-06-19");


UPDATE employee SET super_ssn=333445555 WHERE mname='B';
UPDATE employee SET super_ssn=888665555 WHERE mname='T';
UPDATE employee SET super_ssn=987654321 WHERE mname='J';
UPDATE employee SET super_ssn=888665555 WHERE mname='S';
UPDATE employee SET super_ssn=333445555 WHERE mname='K';
UPDATE employee SET super_ssn=333445555 WHERE mname='A';
UPDATE employee SET super_ssn=987654321 WHERE mname='V';
UPDATE employee SET dno=5 WHERE mname='B';
UPDATE employee SET dno=5 WHERE mname='T';
UPDATE employee SET dno=4 WHERE mname='J';
UPDATE employee SET dno=4 WHERE mname='S';
UPDATE employee SET dno=5 WHERE mname='K';
UPDATE employee SET dno=5 WHERE mname='A';
UPDATE employee SET dno=4 WHERE mname='V';
UPDATE employee SET dno=1 WHERE mname='E';

INSERT INTO dept_locations VALUES(1,"Houston");INSERT INTO dept_locations VALUES(4,"Stafford");INSERT INTO dept_locations VALUES(5,"Bellaire");INSERT INTO dept_locations VALUES(5,"Sugarland");INSERT INTO dept_locations VALUES(5,"Houston");
INSERT INTO project VALUES("ProductX",1,"Bellaire",5);INSERT INTO project VALUES("ProductY",2,"Sugarland",5);INSERT INTO project VALUES("ProductZ",3,"Houston",5);INSERT INTO project VALUES("Computerization",10,"Stafford",4);INSERT INTO project VALUES("Reorganization",20,"Houston",1);INSERT INTO project VALUES("Newbenefits",30,"Stafford",4);
INSERT INTO works_on VALUES(123456789,1,32.5);INSERT INTO works_on VALUES(123456789,2,7.5);INSERT INTO works_on VALUES(666884444,3,40.0);INSERT INTO works_on VALUES(453453453,1,20.0);INSERT INTO works_on VALUES(453453453,2,20.0);INSERT INTO works_on VALUES(333445555,2,10.0);INSERT INTO works_on VALUES(333445555,3,10.0);INSERT INTO works_on VALUES(333445555,10,10.0);INSERT INTO works_on VALUES(333445555,20,10.0);INSERT INTO works_on VALUES(999887777,30,30.0);INSERT INTO works_on VALUES(999887777,10,10.0);INSERT INTO works_on VALUES(987987987,10,35.0);INSERT INTO works_on VALUES(987987987,30,5.0);INSERT INTO works_on VALUES(987654321,30,20.0);INSERT INTO works_on VALUES(987654321,20,15.0);INSERT INTO works_on VALUES(888665555,20,null);
INSERT INTO dependent VALUES(333445555,"Alice",'F',"1986-04-05","Daughter");INSERT INTO dependent VALUES(333445555,"Theodore",'M',"1983-10-25","Son");INSERT INTO dependent VALUES(333445555,"Joy",'F',"1958-05-03","Spouse");INSERT INTO dependent VALUES(987654321,"Abner",'M',"1942-02-28","Spouse");INSERT INTO dependent VALUES(123456789,"Michael",'M',"1988-01-04","Son");INSERT INTO dependent VALUES(123456789,"Alice",'F',"1988-12-30","Daughter");INSERT INTO dependent VALUES(123456789,"Elizabeth",'F',"1967-05-05","Spouse"); 
-----------------------------------------------------------------------------------------------QUERIES-------------------------------------------------------------------------------------------------------------
select * from employee where sex='M' and salary<40000;
select ssn,fname,lname,dno from employee;
select fname,ssn,dno from employee where salary<30000;
select fname,mname,lname,bdate from employee where sex='F';
select fname,lname,address,dependent_name from employee,dependent where employee.ssn=dependent.essn;
select fname,pname from employee,project where employee.dno=project.dnum;
select fname,dlocation from employee,department,dept_locations where dname="Research" and department.dnumber=dept_locations.dnumber and employee.dno=department.dnumber;
select dno,count(ssn),avg(salary) from employee group by(dno);
select sex,max(salary) from employee group by(sex);
select essn,count(dependent_name) from dependent group by(essn);
select essn,sum(hours) from works_on group by(essn);
select pno,sum(hours) from works_on group by(pno);
select e1.fname,e1.lname,e2.fname,e2.lname from employee e1,employee e2 where e1.super_ssn=e2.ssn;