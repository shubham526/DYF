package help;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.Term;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.search.*;
import org.jetbrains.annotations.NotNull;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.*;
import java.util.*;

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



}
