# Java

Author: Sandeep Belgavi. Java 21, Maven. One package per episode under `src/main/java/dev/belgavi/llm/lNN/`.

```
mvn -B -ntp -f java/pom.xml test
mvn -B -ntp -q -f java/pom.xml compile exec:java -Dexec.mainClass=dev.belgavi.llm.l01.TokenCli -Dexec.args="compare data/cache/l01/tokenizers"
```
