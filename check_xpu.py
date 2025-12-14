'''python3 -c "
import os, psycopg2, json
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
# 查询 hits 大于 0 的经验
cur.execute(\"SELECT id, telemetry FROM xpu_entries WHERE (telemetry->>'hits')::int > 0 LIMIT 5;\")
rows = cur.fetchall()
if rows:
    print('🎉 成功！发现有经验被使用了：')
    for row in rows:
        print(f'ID: {row[0]} | 数据: {row[1]}')
else:
    print('⏳ 暂时还没有经验被命中，请继续等待任务运行...')
"'''