<template>
    <div class="task-management-page">
      <el-container>
        <el-aside width="200px">
          <el-menu
            default-active="1"
            class="el-menu-vertical-demo"
            background-color="#545c64"
            text-color="#fff"
            active-text-color="#ffd04b"
          >
            <el-menu-item index="1">
              <el-icon><icon-menu /></el-icon>
              <span>Aufgabenverwaltung</span>
            </el-menu-item>
            <!-- Add more menu items as needed -->
          </el-menu>
        </el-aside>
        <el-container>
          <el-header>
            <h2>Aufgabenverwaltung</h2>
            <el-button type="primary" @click="openAddTaskDialog">Neue Aufgabe</el-button>
          </el-header>
          <el-main>
            <el-row :gutter="20">
              <el-col :span="8">
                <el-card class="task-column">
                  <template #header>
                    <div class="card-header">
                      <span>Offen</span>
                      <el-tag type="info" size="small">{{ todoTasks.length }}</el-tag>
                    </div>
                  </template>
                  <draggable v-model="todoTasks" group="tasks" item-key="id">
                    <template #item="{ element }">
                      <el-card class="task-card" shadow="hover">
                        <h3>{{ element.title }}</h3>
                        <p>{{ element.description }}</p>
                        <div class="task-meta">
                          <el-tag size="small">{{ element.priority }}</el-tag>
                          <span>{{ element.dueDate }}</span>
                        </div>
                      </el-card>
                    </template>
                  </draggable>
                </el-card>
              </el-col>
              <el-col :span="8">
                <el-card class="task-column">
                  <template #header>
                    <div class="card-header">
                      <span>In Bearbeitung</span>
                      <el-tag type="warning" size="small">{{ inProgressTasks.length }}</el-tag>
                    </div>
                  </template>
                  <draggable v-model="inProgressTasks" group="tasks" item-key="id">
                    <template #item="{ element }">
                      <el-card class="task-card" shadow="hover">
                        <h3>{{ element.title }}</h3>
                        <p>{{ element.description }}</p>
                        <div class="task-meta">
                          <el-tag size="small">{{ element.priority }}</el-tag>
                          <span>{{ element.dueDate }}</span>
                        </div>
                      </el-card>
                    </template>
                  </draggable>
                </el-card>
              </el-col>
              <el-col :span="8">
                <el-card class="task-column">
                  <template #header>
                    <div class="card-header">
                      <span>Abgeschlossen</span>
                      <el-tag type="success" size="small">{{ completedTasks.length }}</el-tag>
                    </div>
                  </template>
                  <draggable v-model="completedTasks" group="tasks" item-key="id">
                    <template #item="{ element }">
                      <el-card class="task-card" shadow="hover">
                        <h3>{{ element.title }}</h3>
                        <p>{{ element.description }}</p>
                        <div class="task-meta">
                          <el-tag size="small">{{ element.priority }}</el-tag>
                          <span>{{ element.dueDate }}</span>
                        </div>
                      </el-card>
                    </template>
                  </draggable>
                </el-card>
              </el-col>
            </el-row>
          </el-main>
        </el-container>
      </el-container>
  
      <el-dialog v-model="addTaskDialogVisible" title="Neue Aufgabe" width="30%">
        <el-form :model="newTask" label-width="100px">
          <el-form-item label="Aufgabentitel">
            <el-input v-model="newTask.title"></el-input>
          </el-form-item>
          <el-form-item label="Aufgabenbeschreibung">
            <el-input v-model="newTask.description" type="textarea"></el-input>
          </el-form-item>
          <el-form-item label="Priorität">
            <el-select v-model="newTask.priority">
              <el-option label="Niedrig" value="Niedrig"></el-option>
              <el-option label="Mittel" value="Mittel"></el-option>
              <el-option label="Hoch" value="Hoch"></el-option>
            </el-select>
          </el-form-item>
          <el-form-item label="Fälligkeitsdatum">
            <el-date-picker v-model="newTask.dueDate" type="date" placeholder="Datum auswählen"></el-date-picker>
          </el-form-item>
        </el-form>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="addTaskDialogVisible = false">Abbrechen</el-button>
            <el-button type="primary" @click="addTask">OK</el-button>
          </span>
        </template>
      </el-dialog>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { Menu as IconMenu } from '@element-plus/icons-vue'
  import draggable from 'vuedraggable'
  
  const todoTasks = ref([
    { id: 1, title: 'Neue Funktion entwerfen', description: 'Neue Benutzeroberfläche für die Anwendung entwerfen', priority: 'Hoch', dueDate: '2023-06-15' },
    { id: 2, title: 'Bug beheben', description: 'Von Benutzern gemeldetes Anmeldeproblem lösen', priority: 'Mittel', dueDate: '2023-06-10' },
  ])
  
  const inProgressTasks = ref([
    { id: 3, title: 'Neue API implementieren', description: 'Neue Backend-API entwickeln und testen', priority: 'Hoch', dueDate: '2023-06-20' },
  ])
  
  const completedTasks = ref([
    { id: 4, title: 'Titel', description: 'Titel', priority: 'Niedrig', dueDate: '2023-06-05' },
  ])
  
  const addTaskDialogVisible = ref(false)
  const newTask = ref({
    title: '',
    description: '',
    priority: '',
    dueDate: '',
  })
  
  const openAddTaskDialog = () => {
    addTaskDialogVisible.value = true
  }
  
  const addTask = () => {
    const task = {
      id: Date.now(),
      ...newTask.value,
    }
    todoTasks.value.push(task)
    addTaskDialogVisible.value = false
    newTask.value = {
      title: '',
      description: '',
      priority: '',
      dueDate: '',
    }
  }
  </script>
  
  <style scoped>
  .task-management-page {
    height: 100vh;
  }
  
  .el-aside {
    background-color: #545c64;
  }
  
  .el-header {
    background-color: #fff;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 20px;
  }
  
  .task-column {
    height: calc(100vh - 120px);
    overflow-y: auto;
  }
  
  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .task-card {
    margin-bottom: 10px;
  }
  
  .task-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 10px;
  }
  </style>