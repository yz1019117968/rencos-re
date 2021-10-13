import sys
import lucene
import re
from java.nio.file import Paths
from java.lang import Integer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig,IndexOptions,DirectoryReader
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher, BooleanQuery
from org.apache.lucene.queryparser.classic import QueryParser

lucene.initVM()


def build_index(file_dir):
    """
    Construct the index for Lucene
    :param file_dir:
    :return:
    """
    # 实例化一个SimpleFSDirectory对象，将索引保存至本地文件之中，保存的路径为自定义的路径file_dir+"/lucene_index/"中
    indexDir = SimpleFSDirectory(Paths.get(file_dir+"/lucene_index/"))
    # WhitespaceAnalyzer空格分词，这个分词技术就相当于按照空格简单的切分字符串，对形成的子串不做其他的操作，结果同string.split(" ")的结果类似。
    config = IndexWriterConfig(WhitespaceAnalyzer())
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    # 实例化一个IndexWriter对象，索引写入类。在Directory开辟的储存空间中IndexWriter可以进行索引的写入、修改、增添、删除等操作，但不可进行索引的读取也不能搜索索引。
    writer = IndexWriter(indexDir, config)

    # t1 = FieldType()
    # t1.setStored(True)
    # t1.setTokenized(False)
    # t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
    #
    # t2 = FieldType()
    # t2.setStored(True)
    # t2.setTokenized(True)
    # t2.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    print("%d docs in index" % writer.numDocs())
    if writer.numDocs():
        print("Index already built.")
        return
    with open(file_dir+"/train/train.ast.src") as fc:
        # 读入全部文本
        codes = [re.sub("[\W\s]+|AND|NOT|OR", ' ', line.strip()) for line in fc.readlines()]

    for k, code in enumerate(codes):
        doc = Document()
        # id 字段
        doc.add(StoredField("id", str(k)))
        # code 字段
        # Field.Store是用于表示该域的值是否可以恢复原始字符的变量，Field.Store.YES表示存储在该域中的内容可以恢复至原始文本内容，Field. Store.NOT表示不可恢复。
        doc.add(TextField("code", code, Field.Store.YES))

        writer.addDocument(doc)

    print("Closing index of %d docs..." % writer.numDocs())
    writer.close()


def retriever(file_dir):
    """
    retrieve the most similar code based on ast format, then save the source code and corresponding summaries.
    :param file_dir:
    :return:
    """
    # 分析器
    analyzer = WhitespaceAnalyzer()
    # 读取索引
    reader = DirectoryReader.open(SimpleFSDirectory(Paths.get(file_dir+"/lucene_index/")))
    # 创建索引检索对象
    searcher = IndexSearcher(reader)
    # 采用与建立索引时相同的analyzer
    queryParser = QueryParser("code", analyzer)
    BooleanQuery.setMaxClauseCount(Integer.MAX_VALUE)
    # train.spl.src: split source code; train.txt.tgt: code summaries
    with open(file_dir + "/train/train.spl.src", 'r', encoding="utf-8") as fso,  \
            open(file_dir + "/train/train.txt.tgt", 'r', encoding="utf-8") as fsu:
        sources = [line.strip() for line in fso.readlines()]
        summaries = [line.strip() for line in fsu.readlines()]
    # test.ref.src.0: for storing retrieved source code; ast.out: for storing retrieved summaries.
    with open(file_dir+"/test/test.ast.src", encoding="utf-8") as ft, open(file_dir+"/test/test.ref.src.0", 'w', encoding="utf-8") as fwo, \
            open(file_dir+"/output/ast.out", 'w', encoding="utf-8") as fws:
        queries = [re.sub("[\W\s]+|AND|NOT|OR", ' ', line.strip()) for line in ft.readlines()]

        for i, line in enumerate(queries):
            print("query %d" % i)
            query = queryParser.parse(QueryParser.escape(line))
            # 检索索引，获取符合条件的前1条记录
            hits = searcher.search(query, 1).scoreDocs
            flag = False

            for hit in hits:
                doc = searcher.doc(hit.doc)
                _id = eval(doc.get("id"))
                flag = True
                fwo.write(sources[_id]+'\n')
                fws.write(summaries[_id] + '\n')
            if not flag:
                print(query)
                print(hits)
                exit(-1)


if __name__ == '__main__':
    root = '../samples/%s'%sys.argv[1]
    build_index(root)
    retriever(root)
