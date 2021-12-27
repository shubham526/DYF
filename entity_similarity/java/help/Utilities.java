package help;

import json.AspectLinkExample;
import org.apache.lucene.document.Document;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.*;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;


import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class Utilities {

    @NotNull
    public static Map<String, Set<String>> readEntityFile(String entityFile) {
        Map<String, Set<String>> entityFileMap = new HashMap<>();

        BufferedReader br = null;
        String line , queryID ,entityID;

        try {
            br = new BufferedReader(new FileReader(entityFile));
            while((line = br.readLine()) != null) {
                String[] fields = line.split(" ");
                queryID = fields[0];
                entityID = fields[2];
                Set<String> entitySet = new HashSet<>();
                if(entityFileMap.containsKey(queryID)) {
                    entitySet = entityFileMap.get(queryID);
                }
                entitySet.add(entityID);
                entityFileMap.put(queryID, entitySet);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(br != null) {
                    br.close();
                } else {
                    System.out.println("Buffer has not been initialized!");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return entityFileMap;
    }

    @NotNull
    public static   Map<String, String> readTsvFile(String file) {
        BufferedReader br = null;
        Map<String, String> fileMap = new HashMap<>();
        String line;

        try {
            br = new BufferedReader(new FileReader(file));
            while((line = br.readLine()) != null) {
                String[] fields = line.split("\t");
                if (fields.length == 2) {
                    String key = fields[0];
                    String value = fields[1];
                    fileMap.put(key, value);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(br != null) {
                    br.close();
                } else {
                    System.out.println("Buffer has not been initialized!");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return fileMap;
    }



    /**
     * Write a list of strings to a file in TSV format.
     * @param outFile Path to the output file.
     */

    public static void writeToFile(String outFile, @NotNull List<String> toWrite) {
        BufferedWriter out = null;
        try {
            out = new BufferedWriter(new FileWriter(outFile,true));

            for(String line : toWrite ) {
                out.write(line);
                out.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(out != null) {
                    out.close();
                } else {
                    System.out.println("Buffer has not been initialized!");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }




    @NotNull
    public static String idToText(String id, String field, IndexSearcher searcher) {
        try {
            Document doc = LuceneHelper.searchIndex("Id", id, searcher);
            if (doc != null) {
                return doc
                        .get(field)
                        .replaceAll("\n", " ")
                        .replaceAll("\r", " ");

            }
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        return "";

    }

    @NotNull
    public static List<String> getEntityCategories(String id, IndexSearcher searcher, List<String> stopWords) {
        List<String> categoryNames = new ArrayList<>();
        try {
            Document doc = LuceneHelper.searchIndex("Id", id, searcher);
            if (doc != null) {
                String[] categories = doc.get("CategoryNames").split("\n");
                for (String category : categories) {
                    categoryNames.add(
                            String.join(
                                    " ",
                                    RankingHelper.preProcess(
                                            category.substring(category.indexOf(":")+1).toLowerCase(),
                                            stopWords
                                    )
                            )
                    );
                }
            }
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        return categoryNames;
    }

    @NotNull
    public static String getEntityName(String id, IndexSearcher searcher) {
        try {
            Document doc = LuceneHelper.searchIndex("Id", id, searcher);
            if (doc != null) {
                return doc.get("Title");
            }
        } catch (IOException | ParseException e) {
            e.printStackTrace();
        }
        return "";
    }




    @NotNull
    public static <K, V>LinkedHashMap<K, V> sortByValueDescending(@NotNull Map<K, V> map) {
        LinkedHashMap<K, V> reverseSortedMap = new LinkedHashMap<>();
        map.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue((Comparator<? super V>) Comparator.reverseOrder()))
                .forEachOrdered(x -> reverseSortedMap.put(x.getKey(), x.getValue()));
        return reverseSortedMap;
    }


    /**
     * Reads the stop words file.
     * @param stopWordsFilePath String Path to the stop words file.
     */

    @NotNull
    public static List<String> getStopWords(String stopWordsFilePath) {
        List<String> stopWords = new ArrayList<>();
        BufferedReader br = null;
        String line;

        try {
            br = new BufferedReader(new FileReader(stopWordsFilePath));
            while((line = br.readLine()) != null) {
                stopWords.add(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if(br != null) {
                    br.close();
                } else {
                    System.out.println("Buffer has not been initialized!");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return stopWords;
    }

    @NotNull
    public static  List<AspectLinkExample> readJSONLFile(String filePath) {
        List<AspectLinkExample> aspectLinkExamples = new ArrayList<>();
        try {
            BufferedReader br =
                    new BufferedReader(
                            new InputStreamReader(
                                    new GZIPInputStream(
                                            new FileInputStream(filePath)
                                    )
                            )
                    );

            String jsonLine;
            while ((jsonLine = br.readLine()) != null) {
                JSONObject jsonObject = new JSONObject(jsonLine);
                AspectLinkExample e = new AspectLinkExample(jsonObject);
                aspectLinkExamples.add(e);

            }
        } catch (IOException | JSONException ex) {
            ex.printStackTrace();
        }
        return aspectLinkExamples;
    }

    /**
     * Class to represent an Entity Context Document for an entity.
     * @author Shubham Chatterjee
     * @version 05/31/2020
     */
    public static class EntityContextDocument {

        private final List<Document> documentList;
        private final String entity;
        private final List<String> contextEntities;

        /**
         * Constructor.
         * @param documentList List of documents in the pseudo-document
         * @param entity The entity for which the pseudo-document is made
         * @param contextEntities The list of entities in the pseudo-document
         */
        @Contract(pure = true)
        public EntityContextDocument(List<Document> documentList,
                                     String entity,
                                     List<String> contextEntities) {
            this.documentList = documentList;
            this.entity = entity;
            this.contextEntities = contextEntities;
        }

        /**
         * Method to get the list of documents in the ECD.
         * @return String
         */
        public List<Document> getDocumentList() {
            return this.documentList;
        }

        /**
         * Method to get the entity of the ECD.
         * @return String
         */
        public String getEntity() {
            return this.entity;
        }

        /**
         * Method to get the list of context entities in the ECD.
         * @return ArrayList
         */
        public List<String> getEntityList() {
            return this.contextEntities;
        }
    }

    public static final class Pair<K, V> implements Map.Entry<K, V> {
        private final K key;
        private V value;

        public Pair(K key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public K getKey() {
            return key;
        }

        @Override
        public V getValue() {
            return value;
        }

        @Override
        public V setValue(V value) {
            V old = this.value;
            this.value = value;
            return old;
        }
    }




}
