import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

conn = pymysql.connect(
    host=os.getenv("MYSQL_HOST"),
    port=int(os.getenv("MYSQL_PORT")),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE"),
    charset="utf8mb4"
)
cur = conn.cursor()

tables = {

    # 사용자 계정
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            name        VARCHAR(100)        NOT NULL,
            email       VARCHAR(150) UNIQUE NOT NULL,
            password    VARCHAR(255)        NOT NULL,
            role        ENUM('student','teacher') DEFAULT 'student',
            language    VARCHAR(10)         DEFAULT 'ko',
            created_at  DATETIME            DEFAULT CURRENT_TIMESTAMP
        )
    """,

    # TOPIK 문제 (구조화된 형식)
    "topik_questions": """
        CREATE TABLE IF NOT EXISTS topik_questions (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            exam_no     INT          NOT NULL COMMENT '시험 회차 (예: 102)',
            section     ENUM('listening','reading','writing') NOT NULL,
            question_no INT          NOT NULL COMMENT '문제 번호',
            question    TEXT         NOT NULL,
            option_1    VARCHAR(500),
            option_2    VARCHAR(500),
            option_3    VARCHAR(500),
            option_4    VARCHAR(500),
            answer      TINYINT      COMMENT '정답 번호 (1~4)',
            explanation TEXT         COMMENT '해설',
            level       TINYINT      DEFAULT 2 COMMENT 'TOPIK 급수',
            created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
        )
    """,

    # 퀴즈 결과
    "quiz_results": """
        CREATE TABLE IF NOT EXISTS quiz_results (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            user_id     INT      NOT NULL,
            question_id INT      NOT NULL,
            user_answer TINYINT  COMMENT '사용자가 선택한 답',
            is_correct  TINYINT(1) DEFAULT 0,
            answered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id)     REFERENCES users(id),
            FOREIGN KEY (question_id) REFERENCES topik_questions(id)
        )
    """,

    # 학습 진행도
    "study_progress": """
        CREATE TABLE IF NOT EXISTS study_progress (
            id              INT AUTO_INCREMENT PRIMARY KEY,
            user_id         INT NOT NULL,
            section         ENUM('listening','reading','writing') NOT NULL,
            total_questions INT DEFAULT 0,
            correct_answers INT DEFAULT 0,
            updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """,

    # AI 채팅 히스토리
    "chat_history": """
        CREATE TABLE IF NOT EXISTS chat_history (
            id         INT AUTO_INCREMENT PRIMARY KEY,
            user_id    INT  NOT NULL,
            question   TEXT NOT NULL,
            answer     TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """
}

for name, sql in tables.items():
    cur.execute(sql)
    print(f"✅ {name} 테이블 생성 완료")

conn.commit()
conn.close()
print("\n🎉 모든 테이블 생성 완료!")
