import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/job_status_screen.dart';
import 'screens/templates_screen.dart';
import 'screens/assistant_screen.dart';
import 'screens/editor_screen.dart';
import 'screens/profile_screen.dart';
import 'admin/admin_dashboard.dart';
import 'admin/admin_controls.dart';
import 'admin/admin_users_templates.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env");
  runApp(const VisoraApp());
}

class VisoraApp extends StatelessWidget {
  const VisoraApp({super.key});
  @override Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Visora AI Studio',
      theme: ThemeData(primarySwatch: Colors.deepPurple, scaffoldBackgroundColor: Colors.white),
      debugShowCheckedModeBanner: false,
      initialRoute: '/login',
      routes: {
        '/login': (_) => const LoginScreen(),
        '/home': (_) => const HomeScreen(),
        '/jobStatus': (_) => const JobStatusScreen(),
        '/templates': (_) => const TemplatesScreen(),
        '/assistant': (_) => const AssistantScreen(),
        '/editor': (_) => const EditorScreen(),
        '/profile': (_) => const ProfileScreen(),
        '/admin': (_) => const AdminDashboard(),
        '/admin-controls': (_) => const AdminControls(),
        '/admin-users': (_) => const AdminUsersTemplates(),
      },
    );
  }
}
